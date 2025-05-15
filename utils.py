# utils.py
import openai

def generate_custom_resume(resume_text, job_description, model="gpt-4o"):
    prompt = f"""
You are an expert career coach and resume optimizer. Your task is to rewrite the resume to align perfectly with the job description provided, using action verbs, ATS-friendly keywords, and concise phrasing.

Resume:
{resume_text}

Job Description:
{job_description}

Return ONLY the tailored resume in bullet point format, ready for copy-paste into a PDF generator. Ensure skills and experience are reworded to match the role.
"""

    response = openai.ChatCompletion.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.4
    )

    return response.choices[0].message.content.strip()
