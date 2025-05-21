import streamlit as st
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer, util
from huggingface_hub import InferenceClient
from docx import Document
import PyPDF2
import logging
from retrying import retry

# Download NLTK data
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
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text.strip())
        tokens = word_tokenize(text)
        tokens = [token for token in tokens if token not in self.stop_words]
        return ' '.join(tokens)

    def validate_input(self, resume_text, job_description):
        """Validate input lengths and content."""
        if not resume_text or not job_description:
            raise ValueError("Resume and job description cannot be empty.")
        resume_words = len(resume_text.split())
        job_words = len(job_description.split())
        if resume_words > 2000 or job_words > 500:
            raise ValueError("Input exceeds recommended length. Please shorten the resume or job description.")
        return True

    @retry(stop_max_attempt_number=3, wait_fixed=2000)
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
\"\"\"{job_description}\"\"\"
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
            vectorizer = TfidfVectorizer()
            tfidf_matrix = vectorizer.fit_transform([resume_text, job_description])
            tfidf_score = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            embeddings = self.sentence_model.encode([resume_text, job_description], convert_to_tensor=True)
            semantic_score = util.cos_sim(embeddings[0], embeddings[1]).item()
            combined_score = 0.6 * tfidf_score + 0.4 * semantic_score
            return round(combined_score * 100, 2)
        except Exception as e:
            logger.warning(f"Match score calculation failed: {str(e)}")
            return 0.0

    def extract_resume_text(self, resume_file):
        """Extract text from PDF or DOCX resume."""
        try:
            if resume_file.type == "application/pdf":
                pdf_reader = PyPDF2.PdfReader(resume_file)
                text = "".join(page.extract_text() or "" for page in pdf_reader.pages)
            elif resume_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                doc = Document(resume_file)
                text = "\n".join(paragraph.text for paragraph in doc.paragraphs if paragraph.text.strip())
            else:
                raise ValueError("Unsupported file type. Please upload a PDF or DOCX file.")
            if not text.strip():
                raise ValueError("No text could be extracted from the resume.")
            return text
        except Exception as e:
            logger.error(f"File parsing error: {str(e)}")
            raise ValueError(f"Failed to extract text from resume: {str(e)}")
