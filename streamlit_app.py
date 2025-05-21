import streamlit as st
from utils import ResumeMatcher
from weasyprint import HTML
from io import BytesIO
import markdown

# Page config
st.set_page_config(page_title="AI Resume Customizer", layout="wide")
st.title("🤖 AI Resume Customizer")
st.markdown("Upload your resume (PDF or DOCX) and paste a job description to evaluate fit and download a detailed analysis.")

def generate_pdf(evaluation):
    """Generate PDF from evaluation text using WeasyPrint."""
    # Convert markdown to HTML
    html_content = markdown.markdown(evaluation, extensions=['extra'])
    html_content = f"""
    <html>
    <head>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 1in; }}
            h1 {{ color: #333; }}
            h2 {{ color: #555; }}
            ul {{ margin-left: 20px; }}
        </style>
    </head>
    <body>
        <h1>Resume Evaluation</h1>
        {html_content}
    </body>
    </html>
    """
    pdf_buffer = BytesIO()
    HTML(string=html_content).write_pdf(pdf_buffer)
    return pdf_buffer.getvalue()

def main():
    matcher = ResumeMatcher()
    col1, col2 = st.columns([1, 1])
    with col1:
        resume_file = st.file_uploader("📄 Upload Resume (PDF or DOCX)", type=["pdf", "docx"])
    with col2:
        job_description = st.text_area("📝 Paste Job Description", height=200, help="Enter the job description to compare with your resume.")

    if resume_file and job_description:
        with st.spinner("⏳ Analyzing your resume..."):
            try:
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
    elif resume_file:
        st.warning("⚠️ Please paste the job description to begin analysis.")
    elif job_description:
        st.warning("⚠️ Please upload a resume to begin analysis.")

if __name__ == "__main__":
    main()
