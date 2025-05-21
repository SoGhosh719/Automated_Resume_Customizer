import streamlit as st
from huggingface_hub import InferenceClient
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

client = InferenceClient(
    model="HuggingFaceH4/zephyr-7b-beta",
    token=st.secrets["HUGGINGFACE_TOKEN"]
)

def generate_custom_resume(resume_text, job_description):
    system_prompt = (
        "You are an expert resume writer. Your task is to rewrite the user's resume "
        "to match the provided job description using action verbs, bullet points, and keywords "
        "from the job. Make the resume ATS-friendly and role-specific. Format the output professionally "
        "as a list of resume bullet points only."
    )

    user_prompt = f"""
Here is a resume and a job description.

Resume:
\"\"\"
{resume_text}
\"\"\"

Job Description:
\"\"\"
{job_description}
\"\"\"

Rewrite the resume so it aligns with the job. Return only the improved resume in bullet points.
"""

    try:
        response = client.chat_completion(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=600,
            temperature=0.4
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"⚠️ Hugging Face API error: {str(e)}"

def compute_match_score(resume_text, job_description):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([resume_text, job_description])
    score = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
    return round(score * 100, 2)
