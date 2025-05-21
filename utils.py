import streamlit as st
from huggingface_hub import InferenceClient
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Initialize Hugging Face client
client = InferenceClient(
    model="HuggingFaceH4/zephyr-7b-beta",
    token=st.secrets["HUGGINGFACE_TOKEN"]
)

def review_resume_for_job_fit(resume_text, job_description):
    """
    Uses Zephyr to evaluate how well the resume aligns with the job description.
    Returns structured feedback with overall fit, strengths, gaps, and suggestions.
    """
    system_prompt = (
        "You are a professional career coach and resume reviewer. Your task is to evaluate how well a candidate’s resume "
        "aligns with a specific job description. Identify strengths, gaps, and suggest what should be added or changed. "
        "Do not rewrite the resume. Provide clear bullet points under each section."
    )

    user_prompt = f"""
Below is a candidate's resume and the job description they are applying for.

Resume:
\"\"\"{resume_text}\"\"\"

Job Description:
\"\"\"{job_description}\"\"\"

Please evaluate the resume in relation to the job and return the following:

### 1. OVERALL FIT
(A brief paragraph summarizing the match)

### 2. STRENGTHS
- Bullet points listing aspects of the resume that are well-aligned with the job

### 3. GAPS / MISSING INFORMATION
- Bullet points listing key responsibilities, qualifications, or tools from the job description that are missing or underemphasized

### 4. SUGGESTIONS
- Specific, actionable suggestions for improving the resume to better align with the job

Respond only in the structure above.
"""

    try:
        response = client.chat_completion(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=900,
            temperature=0.3
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"⚠️ Hugging Face API error:\n{str(e)}"

def compute_match_score(resume_text, job_description):
    """
    Computes cosine similarity-based match score using TF-IDF vectors.
    """
    try:
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform([resume_text, job_description])
        score = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        return round(score * 100, 2)
    except Exception as e:
        st.warning(f"⚠️ Match score calculation failed: {str(e)}")
        return 0.0
