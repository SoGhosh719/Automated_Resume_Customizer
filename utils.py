import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os

FIREWORKS_API_KEY = os.getenv("FIREWORKS_API_KEY")

def generate_custom_resume(resume_text, job_description):
    prompt = f"""You are an expert career coach and resume optimizer.
Your task is to rewrite the resume to align with the job description provided, using action verbs, ATS-friendly keywords, and concise phrasing.

Resume:
{resume_text}

Job Description:
{job_description}

Return ONLY the tailored resume in bullet point format.
"""

    headers = {
        "Authorization": f"Bearer {FIREWORKS_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "accounts/fireworks/models/mixtral-8x7b-instruct",  # Public hosted
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.4
    }

    try:
        response = requests.post(
            "https://api.fireworks.ai/inference/v1/chat/completions",
            json=payload,
            headers=headers
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"⚠️ Error contacting Fireworks AI: {str(e)}"

def compute_match_score(resume_text, job_description):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([resume_text, job_description])
    score = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
    return score * 100
