import streamlit as st
from utils import ResumeMatcher
from fpdf import FPDF
from io import BytesIO
import unicodedata

# Page config
st.set_page_config(page_title="AI Resume Customizer", layout="wide")
st.title("🤖 AI Resume Customizer")
st.markdown("Upload your resume (PDF or DOCX) and paste a job description to evaluate fit and download a detailed analysis.")

def clean_unicode(text):
    """Normalize Unicode characters to ASCII."""
    return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")

def generate_pdf(evaluation):
    """Generate PDF from evaluation text using FPDF."""
    cleaned_text = clean_unicode(evaluation)
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_font("Arial", size=11)
    for line in cleaned_text.split("\n"):
        pdf.multi_cell(0, 10, line)
    pdf_bytes = pdf.output(dest='S').encode('latin-1')
    return BytesIO(pdf_bytes)

def main():
    matcher = ResumeMatcher()
    col1, col2 = st.columns([1, 1])
    with col1:
        resume_file = st.file_uploader("📄 Upload Resume (PDF or DOCX)", type=["pdf", "docx"])
    with col2:
        job_description = st.text_area("📝 Paste Job Description", height=200, help="Enter the job description (max 1500 words).")

    if resume_file:
        try:
            resume_text = matcher.extract_resume_text(resume_file)
            resume_word_count = len(resume_text.split())
            st.markdown(f"**Resume Word Count**: {resume_word_count}/5000")
            with st.expander("Preview Extracted Resume Text"):
                st.text_area("Resume Text", resume_text, height=150, disabled=True)
        except ValueError as ve:
            st.error(f"❌ {str(ve)}")
            return

    if job_description:
        job_word_count = len(job_description.split())
        st.markdown(f"**Job Description Word Count**: {job_word_count}/1500")

    if resume_file and job_description:
        with st.spinner("⏳ Analyzing your resume..."):
            try:
                # Additional job description validation
                if len(job_description.strip()) < 50:
                    st.error("❌ Job description is too short. Please provide at least 50 characters.")
                    return
                resume_text = matcher.extract_resume_text(resume_file)
                matcher.validate_input(resume_text, job_description)
                score = matcher.compute_match_score(resume_text, job_description)
                st.markdown("### 🌟 Resume vs Job Description Match")
                st.metric(label="Match Score", value=f"{score:.2f}%", help="Higher scores indicate better alignment.")
                evaluation = matcher.review_resume_for_job_fit(resume_text, job_description)
                st.markdown("### 🧠 AI Evaluation of Resume Fit")
                st.markdown(evaluation)
                pdf_data = generate_pdf(evaluation)
                st.download_button(
                    label="📥 Download Evaluation as PDF",
                    data=pdf_data,
                    file_name="Resume_Evaluation.pdf",
                    mime="application/pdf"
                )
            except ValueError as ve:
                st.error(f"❌ {str(ve)}")
            except Exception as e:
                st.error("❌ An unexpected error occurred during analysis. Please try again.")
                st.exception(e)

    elif resume_file:
        st.warning("⚠️ Please paste the job description to begin analysis.")
    elif job_description:
        st.warning("⚠️ Please upload a resume to begin analysis.")

if __name__ == "__main__":
    main()
