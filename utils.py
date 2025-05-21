import streamlit as st
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer, util
from huggingface_hub import InferenceClient
import uuid
import retrying
import logging

# Download NLTK data (run once in the environment)
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ResumeMatcher:
    def __init__(self, model_name="HuggingFaceH4/zephyr-7b-beta", max_tokens=900, temperature=0.3):
        """Initialize the ResumeMatcher with configurable parameters."""
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.client = None
        self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.stop_words = set(stopwords.words('english'))
        try:
            self.client = InferenceClient(model=self.model_name, token=st.secrets["HUGGINGFACE_TOKEN"])
        except Exception as e:
            st.error(f"Failed to initialize Hugging Face client: {str(e)}")
            logger.error(f"Client initialization error: {str(e)}")

    def clean_text(self, text):
        """Clean and preprocess text for analysis."""
        if not text or not isinstance(text, str):
            return ""
        # Convert to lowercase, remove special characters, and normalize whitespace
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text.strip())
        # Tokenize and remove stop words
        tokens = word_tokenize(text)
        tokens = [token for token in tokens if token not in self.stop_words]
        return ' '.join(tokens)

    def validate_input(self, resume_text, job_description):
        """Validate input lengths and content."""
        if not resume_text or not job_description:
            raise ValueError("Resume and job description cannot be empty.")
        # Check approximate token count (rough estimate: 1 word ~ 1.3 tokens)
        resume_words = len(resume_text.split())
        job_words = len(job_description.split())
        if resume_words > 2000 or job_words > 500:
            raise ValueError("Input exceeds recommended length. Please shorten the resume or job description.")
        return True

    @retrying.retry(stop_max_attempt_number=3, wait_fixed=2000)
    def review_resume_for_job_fit(self, resume_text, job_description):
        """Evaluate resume fit using Hugging Face API with retry logic."""
        if not self.client:
            raise RuntimeError("Hugging Face client not initialized.")

        system_prompt = (
            "You are a professional career coach and resume reviewer. Your task is to evaluate how well a candidate’s resume "
            "aligns with a specific job description. Identify strengths, gaps, and suggest what should be added or changed. "
            "Do not rewrite the resume. Provide clear bullet points under each section."
        )

        user_prompt = f"""
Resume:
\"\"\"{resume_text}\"\"\"

Job Description:
\"\"\"{job_description}\"\"

### 1. OVERALL FIT
(A brief paragraph summarizing the match)

### 2. STRENGTHS
- Bullet points listing aspects of the resume that are well-aligned with the job

### 3. GAPS / MISSING INFORMATION
- Bullet points listing key responsibilities, qualifications, or tools from the job description that are missing or underemphasized

### 4. SUGGESTIONS
- Specific, actionable suggestions for improving the resume to better align with the job
"""

        try:
            response = self.client.chat_completion(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=self.max_tokens,
                temperature=self.temperature
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"API call failed: {str(e)}")
            raise

    def compute_match_score(self, resume_text, job_description):
        """Compute match score using TF-IDF and semantic similarity."""
        try:
            resume_text = self.clean_text(resume_text)
            job_description = self.clean_text(job_description)

            # TF-IDF similarity
            vectorizer = TfidfVectorizer()
            tfidf_matrix = vectorizer.fit_transform([resume_text, job_description])
            tfidf_score = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]

            # Semantic similarity using sentence transformers
            embeddings = self.sentence_model.encode([resume_text, job_description], convert_to_tensor=True)
            semantic_score = util.cos_sim(embeddings[0], embeddings[1]).item()

            # Combine scores (weighted average)
            combined_score = 0.6 * tfidf_score + 0.4 * semantic_score
            return round(combined_score * 100, 2)
        except Exception as e:
            logger.warning(f"Match score calculation failed: {str(e)}")
            return 0.0

    def run(self):
        """Run the Streamlit application."""
        st.title("Resume Matcher")
        st.markdown("Upload your resume and job description to evaluate their fit.")

        # File upload for resume
        resume_file = st.file_uploader("Upload Resume (TXT or PDF)", type=["txt", "pdf"])
        resume_text = ""
        if resume_file:
            if resume_file.type == "text/plain":
                resume_text = resume_file.read().decode("utf-8")
            elif resume_file.type == "application/pdf":
                try:
                    import PyPDF2
                    pdf_reader = PyPDF2.PdfReader(resume_file)
                    resume_text = "".join(page.extract_text() for page in pdf_reader.pages)
                except Exception as e:
                    st.error(f"Failed to read PDF: {str(e)}")

        # Text input for job description
        job_description = st.text_area("Paste Job Description", height=200)

        if st.button("Analyze"):
            try:
                self.validate_input(resume_text, job_description)
                with st.spinner("Analyzing resume..."):
                    # Compute match score
                    match_score = self.compute_match_score(resume_text, job_description)
                    st.subheader("Match Score")
                    st.write(f"**{match_score}%** (Higher scores indicate better alignment)")

                    # Get qualitative review
                    review = self.review_resume_for_job_fit(resume_text, job_description)
                    st.subheader("Resume Review")
                    st.markdown(review)
            except ValueError as ve:
                st.error(str(ve))
            except Exception as e:
                st.error("An error occurred during analysis. Please try again.")

if __name__ == "__main__":
    matcher = ResumeMatcher()
    matcher.run()
