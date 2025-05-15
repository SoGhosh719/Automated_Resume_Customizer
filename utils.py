import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Sends prompts to local Ollama model (http://localhost:11434)
def generate_custom_resume(resume_text, job_description):
    prompt = f"""
You are an expert career coach and resume optimizer. Your task is to rewrite the resume to align with the job description provided, using action verbs, ATS-friendly keywords, and concise phrasing.

Resume:
{resume_text}

Job Description:
{job_description}

Return ONLY the tailored resume in bullet point format.
"""
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": "mistral", "prompt": prompt, "stream": False}
        )
        result = response.json()
        return result.get("response", "").strip()
    except Exception as e:
        return f"⚠️ Error calling local Mistral model: {str(e)}"

def compute_match_score(resume_text, job_description):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([resume_text, job_description])
    score = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
    return score * 100
