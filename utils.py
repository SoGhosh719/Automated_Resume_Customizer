import openai
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

client = openai.OpenAI()  # Uses your env variable OPENAI_API_KEY

def generate_custom_resume(resume_text, job_description, model="gpt-4o"):
    prompt = f"""
You are an expert career coach and resume optimizer. Your task is to rewrite the resume to align with the job description provided, using action verbs, ATS-friendly keywords, and concise phrasing.

Resume:
{resume_text}

Job Description:
{job_description}

Return ONLY the tailored resume in bullet point format.
"""
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.4
    )
    return response.choices[0].message.content.strip()

def compute_match_score(resume_text, job_description):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([resume_text, job_description])
    score = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
    return score * 100
