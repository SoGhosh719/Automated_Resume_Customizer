import streamlit as st
from resume_parser import extract_resume_text
from utils import review_resume_for_job_fit, compute_match_score
from fpdf import FPDF
import unicodedata
from io import BytesIO
import traceback

# ---------- PAGE CONFIG ----------
st.set_page_config(page_title="AI Resume Customizer", layout="wide")
st.title("🤖 Automated Resume Tailor (Hugging Face Model)")
st.write("Upload your resume and a job description to get a customized version and a match score.")

# ---------- FILE UPLOAD ----------
resume_file = st.file_uploader("📄 Upload Resume (PDF or DOCX)", type=["pdf", "docx"])
job_description = st.text_area("📝 Paste the Job Description")

# ---------- HELPER: UNICODE CLEANER ----------
def clean_unicode(text):
    return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")

# ---------- HELPER: STREAMLIT PDF DOWNLOAD BUTTON ----------
def get_pdf_download_button(text, filename):
    cleaned_text = clean_unicode(text)
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_font("Arial", size=11)

    for line in cleaned_text.split("\n"):
        pdf.multi_cell(0, 10, line)

    pdf_bytes = pdf.output(dest='S').encode('latin-1')  # Fix: Generate PDF as string
    buffer = BytesIO(pdf_bytes)

    st.download_button(
        label="📥 Download Evaluation as PDF",
        data=buffer,
        file_name=filename,
        mime="application/pdf"
    )

# ---------- MAIN APP LOGIC ----------
if resume_file and job_description:
    with st.spinner("⏳ Analyzing your resume..."):
        try:
            resume_text = extract_resume_text(resume_file)

            # Get LLM-based review
            evaluation = review_resume_for_job_fit(resume_text, job_description)

            # Get TF-IDF score
            score = compute_match_score(resume_text, job_description)

            st.markdown("### 🌟 Resume vs Job Description Match")
            st.metric(label="Match Score", value=f"{score:.2f}%")

            st.markdown("### 🧠 AI Evaluation of Resume Fit")
            st.markdown(evaluation)

            get_pdf_download_button(evaluation, "Resume_Evaluation.pdf")

        except Exception as e:
            st.error("❌ An unexpected error occurred during processing.")
            st.text(traceback.format_exc())

elif resume_file and not job_description:
    st.warning("⚠️ Please paste the job description to begin analysis.")
elif job_description and not resume_file:
    st.warning("⚠️ Please upload a resume to begin analysis.")
