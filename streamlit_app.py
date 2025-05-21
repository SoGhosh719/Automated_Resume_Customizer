import streamlit as st
from utils import ResumeMatcher
import re

# Page config
st.set_page_config(page_title="AI Resume Customizer", layout="wide")
st.title("🤖 AI Resume Customizer")
st.markdown("Upload your resume (PDF or DOCX) and paste a job description to evaluate fit and download a detailed analysis.")

def generate_latex_content(evaluation):
    """Generate LaTeX code for PDF export."""
    # Escape special characters for LaTeX
    evaluation = re.sub(r'([#$%&_{}\\])', r'\\\1', evaluation)
    evaluation = evaluation.replace('~', r'\textasciitilde{}').replace('^', r'\textasciicircum{}')
    latex_content = r"""
\documentclass[a4paper,11pt]{article}
\usepackage{geometry}
\geometry{margin=1in}
\usepackage{enumitem}
\usepackage{parskip}
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage{lmodern}
\begin{document}
\section*{Resume Evaluation}
""" + "\n".join(line if not line.startswith('#') else r"\section*{" + line.replace('#', '').strip() + "}" for line in evaluation.split("\n")) + r"""
\end{document}
"""
    return latex_content

def main():
    matcher = ResumeMatcher()
    col1, col2 = st.columns([1, 1])
    with col1:
        resume_file = st.file_uploader("📄 Upload Resume (PDF or DOCX)", type=["pdf", "docx"])
    with col2:
        job_description = st.text_area("📝 Paste Job Description", height=200)

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
                latex_content = generate_latex_content(evaluation)
                st.download_button(
                    label="📥 Download Evaluation as LaTeX",
                    data=latex_content,
                    file_name="Resume_Evaluation.tex",
                    mime="text/latex"
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
