import streamlit as st
from huggingface_hub import InferenceClient
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Initialize the Hugging Face client with Zephyr
client = InferenceClient(
    model="HuggingFaceH4/zephyr-7b-beta",
    token=st.secrets["HUGGINGFACE_TOKEN"]
)

def generate_custom_resume(resume_text, job_description):
    system_prompt = (
        "You are a professional resume writer and career strategist. Your goal is to rewrite the candidate’s resume "
        "to maximize alignment with the given job description. Use clear, compelling bullet points, embed keywords from the job, "
        "and maintain a professional tone suitable for applicant tracking systems (ATS). Focus on role-relevant achievements, "
        "skills, and experiences without copying the job description directly. Format the output in ATS-friendly bullet points."
    )

    user_prompt = f"""
The following content includes a candidate's resume and the job description for the role they are applying to.

Resume:
\"\"\"
{resume_text}
\"\"\"

Job Description:
\"\"\"
{job_description}
\"\"\"

Please rewrite the resume to emphasize how the candidate’s experience aligns with the role. Output only the rewritten resume in bullet-point format with no introductory or closing remarks.
"""

    try:
        response = client.chat_completion(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=700,
            temperature=0.3
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"⚠️ Hugging Face API error: {str(e)}"

def compute_match_score(resume_text, job_description):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([resume_text, job_description])
    score = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
    return round(score * 100, 2)
