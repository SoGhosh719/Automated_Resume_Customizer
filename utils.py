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
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class ResumeMatcher:
    def __init__(self, model_name="mistralai/Mixtral-8x7B-Instruct-v0.1", max_tokens=2000, temperature=0.3):
        """Initialize the ResumeMatcher with configurable parameters."""
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.client = None
        self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.stop_words = set(stopwords.words('english'))
        try:
            self.client = InferenceClient(model=self.model_name, token=st.secrets["HUGGINGFACE_TOKEN"])
            logger.debug("Hugging Face client initialized successfully")
        except Exception as e:
            st.error(f"Failed to initialize Hugging Face client: {str(e)}")
            logger.error(f"Client initialization error: {str(e)}", exc_info=True)

    def clean_text(self, text):
        """Clean and preprocess text for analysis."""
        if not text or not isinstance(text, str):
            logger.warning("Empty or invalid text input for cleaning")
            return ""
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text.strip())
        tokens = word_tokenize(text)
        tokens = [token for token in tokens if token not in self.stop_words]
        cleaned_text = ' '.join(tokens)
        logger.debug(f"Cleaned text: {len(cleaned_text.split())} words")
        return cleaned_text

    def validate_input(self, resume_text, job_description):
        """Validate input lengths and content."""
        if not resume_text or not job_description:
            raise ValueError("Resume and job description cannot be empty.")
        resume_words = len(resume_text.split())
        job_words = len(job_description.split())
        if resume_words > 5000 or job_words > 1500:
            raise ValueError(
                f"Input exceeds recommended length (resume: {resume_words}/5000 words, "
                f"job description: {job_words}/1500 words). Please shorten the input."
            )
        logger.debug(f"Input validated: resume={resume_words} words, job_description={job_words} words")
        return True

    def truncate_text(self, text, max_words=1500):
        """Truncate text to a maximum number of words."""
        words = text.split()
        if len(words) > max_words:
            logger.warning(f"Truncating text from {len(words)} to {max_words} words")
            return ' '.join(words[:max_words])
        return text

    @retry(stop_max_attempt_number=3, wait_fixed=2000)
    def review_resume_for_job_fit(self, resume_text, job_description):
        """Evaluate resume fit using Hugging Face API with retry logic."""
        if not self.client:
            raise RuntimeError("Hugging Face client not initialized.")
        logger.debug("Preparing API call for resume review")
        # Truncate inputs to prevent context overflow
        resume_text = self.truncate_text(resume_text, max_words=1500)
        job_description = self.truncate_text(job_description, max_words=800)
        system_prompt = (
            "You are a professional career coach and resume reviewer. Your task is to evaluate how well a candidate’s resume "
            "aligns with a job description. Compare the resume’s skills (technical and soft), work experience, education, and "
            "certifications against the job’s requirements and responsibilities. Prioritize key qualifications, responsibilities, "
            "and tools explicitly mentioned in the job description (e.g., data protection, financial management). Focus on the "
            "most relevant resume sections (e.g., recent experience, key skills) and avoid fabricating details not present in the "
            "resume. If inputs are lengthy, summarize relevant sections. Provide a structured response with exactly the following "
            "sections, and do not deviate from this format:\n\n"
            "### 1. OVERALL FIT\n"
            "A concise paragraph (3-5 sentences) summarizing the degree of alignment between the resume and job description. "
            "Highlight the strongest areas of match and any critical gaps.\n\n"
            "### 2. STRENGTHS\n"
            "- Bullet points listing specific skills, experiences, education, or certifications in the resume that closely align "
            "with the job’s requirements.\n"
            "- Include examples or details from the resume that demonstrate a strong match (e.g., specific tools, projects, or achievements).\n\n"
            "### 3. GAPS / MISSING INFORMATION\n"
            "- Bullet points identifying key responsibilities, qualifications, skills, or tools from the job description that are "
            "missing or insufficiently addressed in the resume.\n"
            "- Note if any gaps are due to unclear or incomplete inputs.\n\n"
            "### 4. SUGGESTIONS\n"
            "- Specific, actionable recommendations to improve the resume’s alignment with the job description.\n"
            "- Suggest adding, rephrasing, or emphasizing specific skills, experiences, or keywords.\n"
            "- Recommend quantifying achievements (e.g., 'increased sales by 20%') where applicable.\n"
            "- Ensure suggestions are practical and tailored to the candidate’s background.\n\n"
            "Do not include any other sections or deviate from this structure. Avoid generating information not explicitly present in the resume."
        )
        user_prompt = f"""
Resume:
\"\"\"{resume_text}\"\"\"
Job Description:
\"\"\"{job_description}\"\"\"
"""
        try:
            logger.debug(f"Sending API request with {len(resume_text.split())} resume words and {len(job_description.split())} job description words")
            response = self.client.chat_completion(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=self.max_tokens,
                temperature=self.temperature
            )
            logger.debug("API response received successfully")
            response_text = response.choices[0].message.content.strip()
            # Validate response structure
            required_sections = ["### 1. OVERALL FIT", "### 2. STRENGTHS", "### 3. GAPS / MISSING INFORMATION", "### 4. SUGGESTIONS"]
            if not all(section in response_text for section in required_sections):
                logger.warning("API response missing required sections")
                return f"Error: Incomplete API response. Match score: {self.compute_match_score(resume_text, job_description):.2f}%"
            return response_text
        except Exception as e:
            logger.error(f"API call failed: {str(e)}", exc_info=True)
            raise

    @st.cache_data
    def compute_match_score(_self, resume_text, job_description):
        """Compute match score using TF-IDF and semantic similarity."""
        try:
            resume_text = _self.clean_text(resume_text)
            job_description = _self.clean_text(job_description)
            if not resume_text or not job_description:
                logger.error("Empty text after cleaning for match score")
                return 0.0
            vectorizer = TfidfVectorizer()
            tfidf_matrix = vectorizer.fit_transform([resume_text, job_description])
            tfidf_score = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            embeddings = _self.sentence_model.encode([resume_text, job_description], convert_to_tensor=True)
            semantic_score = util.cos_sim(embeddings[0], embeddings[1]).item()
            combined_score = 0.6 * tfidf_score + 0.4 * semantic_score
            logger.debug(f"Match score computed: TF-IDF={tfidf_score:.2f}, Semantic={semantic_score:.2f}, Combined={combined_score:.2f}")
            return round(combined_score * 100, 2)
        except Exception as e:
            logger.error(f"Match score calculation failed: {str(e)}", exc_info=True)
            return 0.0

    def extract_resume_text(self, resume_file):
        """Extract text from PDF or DOCX resume with file size validation."""
        try:
            # Validate file size (max 5MB)
            if resume_file.size > 5 * 1024 * 1024:
                raise ValueError("File size exceeds 5MB limit.")
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
            logger.debug(f"Resume text extracted: {len(text.split())} words")
            return text
        except Exception as e:
            logger.error(f"File parsing error: {str(e)}", exc_info=True)
            raise ValueError(f"Failed to extract text from resume: {str(e)}")
