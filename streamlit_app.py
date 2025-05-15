import streamlit as st
import PyPDF2
import openai
from io import BytesIO

st.title("🤖 Automated Resume Customizer")
st.write("Upload your resume and a job description to get a tailored version.")

# Uploads
resume_file = st.file_uploader("Upload Your Resume (PDF)", type=["pdf"])
job_description = st.text_area("Paste Job Description")

# GPT Prompt Template
def generate_custom_resume(resume_text, jd_text):
    prompt = f"""You are a career coach and resume expert.
Given the following resume and job description, tailor the resume to maximize keyword match and relevance.

Resume:
{resume_text}

Job Description:
{jd_text}

Return the improved resume text in a professional tone."""

    response = openai.ChatCompletion.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.4
    )
    return response.choices[0].message.content

# Extract PDF text
def extract_text_from_pdf(file):
    pdf = PyPDF2.PdfReader(file)
    text = ""
    for page in pdf.pages:
        text += page.extract_text()
    return text

if resume_file and job_description:
    with st.spinner("Tailoring your resume..."):
        resume_text = extract_text_from_pdf(resume_file)
        customized_resume = generate_custom_resume(resume_text, job_description)
        st.text_area("Customized Resume", value=customized_resume, height=600)
