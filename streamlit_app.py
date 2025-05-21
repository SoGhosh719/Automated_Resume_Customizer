import streamlit as st
from resume_parser import extract_resume_text
from utils import generate_custom_resume, review_resume_for_job_fit, compute_match_score
from fpdf import FPDF
import unicodedata
from io import BytesIO

# ---------- PAGE CONFIG ----------
st.set_page_config(page_title="AI Resume Assistant", layout="wide")
st.title("🧠 Smart Resume Tailor & Review (Fireworks + Mixtral)")
st.write("Upload your resume and a job description to get a tailored version, match score, and detailed feedback.")

# ---------- FILE UPLOAD ----------
resume_file = st.file_uploader("📄 Upload Resume (PDF or DOCX)", type=["pdf", "docx"])
job_description = st.text_area("📝 Paste the Job Description")

# ---------- MODE SWITCH ----------
mode = st.radio("Choose Mode", ["Tailor Resume", "Review Resume Fit"])

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

    buffer = BytesIO()
    pdf.output(buffer)
    buffer.seek(0)

    st.download_button(
        label="📅 Download Output as PDF",
        data=buffer,
        file_name=filename,
        mime="application/pdf"
    )

# ---------- MAIN APP LOGIC ----------
if resume_file and job_description:
    with st.spinner("⏳ Processing your resume..."):
        resume_text = extract_resume_text(resume_file)
        match_score = compute_match_score(resume_text, job_description)

        if mode == "Tailor Resume":
            output_text = generate_custom_resume(resume_text, job_description)
            title = "📝 Customized Resume"
            filename = "Customized_Resume.pdf"
        else:
            output_text = review_resume_for_job_fit(resume_text, job_description)
            title = "📒 Resume Review Feedback"
            filename = "Resume_Review_Feedback.pdf"

        if output_text.startswith("⚠️ Error"):
            st.error(output_text)
        else:
            st.markdown("### 🌟 Match Score")
            st.metric(label="Resume vs JD Match", value=f"{match_score:.2f}%")

            st.markdown(f"### {title}")
            st.text_area("Output", value=output_text, height=500)
            get_pdf_download_button(output_text, filename)

elif resume_file and not job_description:
    st.warning("Please paste the job description to begin analysis.")
elif job_description and not resume_file:
    st.warning("Please upload a resume to begin analysis.")
