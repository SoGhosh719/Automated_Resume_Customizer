import streamlit as st
from huggingface_hub import InferenceClient
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ✅ Pull token from Streamlit secrets
client = InferenceClient(
    model="HuggingFaceH4/zephyr-7b-beta",
    token=st.secrets["HUGGINGFACE_TOKEN"]
)

def generate_custom_resume(resume_text, job_description):
    prompt = f"""You are an expert resume writer. Rewrite the following resume to fit the job description. Use bullet points, action verbs, and align it with the job role.

Resume:
{resume_text}

Job Description:
{job_description}

Output only the tailored resume in bullet format.
"""

    try:
        response = client.text_generation(
            prompt=prompt,
            max_new_tokens=500,
            temperature=0.4
        )
        return response.strip()
    except Exception as e:
        return f"⚠️ Hugging Face API error: {str(e)}"

def compute_match_score(resume_text, job_description):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([resume_text, job_description])
    score = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
    return round(score * 100, 2)
