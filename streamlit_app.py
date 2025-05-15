import streamlit as st
from resume_parser import extract_resume_text
from utils import generate_custom_resume, compute_match_score
from fpdf import FPDF
import base64

st.set_page_config(page_title="AI Resume Customizer", layout="wide")
st.title("🤖 Automated Resume Tailor with GPT-4o")
st.write("Upload your resume and a job description to get a customized version and match score.")

resume_file = st.file_uploader("📄 Upload Resume (PDF or DOCX)", type=["pdf", "docx"])
job_description = st.text_area("📝 Paste the Job Description")

def create_download_link(text, filename):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_font("Arial", size=11)
    for line in text.split("\n"):
        pdf.multi_cell(0, 10, line)
    pdf_output_path = f"/mnt/data/{filename}"
    pdf.output(pdf_output_path)
    with open(pdf_output_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    return f'<a href="data:application/octet-stream;base64,{b64}" download="{filename}">📥 Download Customized Resume</a>'

if resume_file and job_description:
    with st.spinner("⏳ Tailoring your resume..."):
        resume_text = extract_resume_text(resume_file)
        tailored_resume = generate_custom_resume(resume_text, job_description)
        score = compute_match_score(resume_text, job_description)

        st.markdown("### 🎯 Match Score")
        st.metric(label="Resume vs JD Match", value=f"{score:.2f}%")

        st.markdown("### 📝 Customized Resume")
        st.text_area("Output", value=tailored_resume, height=500)

        download_link = create_download_link(tailored_resume, "Customized_Resume.pdf")
        st.markdown(download_link, unsafe_allow_html=True)
