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
        "You are a senior resume writer and career strategist with expertise in ATS optimization. "
        "Your job is to rewrite resumes to align closely with job descriptions using structured sections. "
        "Incorporate relevant keywords, tailor experience to reflect the job’s requirements, and format the resume clearly. "
        "Avoid copying the job description or introducing fluff."
    )

    user_prompt = f"""
Below is a candidate's original resume and a job description they are applying for.

Resume:
\"\"\"
{resume_text}
\"\"\"

Job Description:
\"\"\"
{job_description}
\"\"\"

Rewrite the resume so that it is aligned with the job and presented in the following structured format:

### PROFESSIONAL SUMMARY
(A 2–3 line overview of the candidate's suitability for the role)

### KEY SKILLS
- Bullet points of technical and soft skills aligned with the job description

### EXPERIENCE
- Rewritten bullet points based on the candidate's experience, highlighting responsibilities and achievements relevant to the job

### EDUCATION
- Keep the most relevant academic background or certifications

Output only the rewritten resume in this structure with no extra commentary.
"""

    try:
        response = client.chat_completion(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=900,
            temperature=0.25
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"⚠️ Hugging Face API error: {str(e)}"

def compute_match_score(resume_text, job_description):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([resume_text, job_description])
    score = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
    return round(score * 100, 2)
