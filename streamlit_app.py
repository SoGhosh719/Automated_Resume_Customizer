import streamlit as st
from resume_parser import extract_resume_text
from utils import review_resume_for_job_fit, compute_match_score

st.set_page_config(page_title="Resume Fit Review", layout="wide")
st.title("🧠 AI Resume Fit Review Assistant")

st.markdown("""
Welcome! This app helps you improve your resume by:
- 🔍 Comparing your resume to a specific job description
- 📋 Giving structured feedback on alignment, strengths, and gaps
- 🎯 Suggesting improvements to boost your chances
""")

# Upload and input
uploaded_file = st.file_uploader("📄 Upload your resume (.pdf or .docx)", type=["pdf", "docx"])
job_description = st.text_area("📝 Paste the job description here")

# Feature toggle
mode = st.radio("Choose Mode", ["Review Resume Fit", "Tailor Resume (Coming Soon)"])

if uploaded_file and job_description:
    with st.spinner("🔍 Analyzing your resume..."):
        resume_text = extract_resume_text(uploaded_file)
        review_feedback = review_resume_for_job_fit(resume_text, job_description)
        match_score = compute_match_score(resume_text, job_description)

    st.success("✅ Analysis Complete!")
    
    st.subheader("🎯 Resume-Job Match Score")
    st.metric("Match Score", f"{match_score:.2f}%")

    st.subheader("🧾 Resume Review Feedback")
    st.text_area("What to improve", value=review_feedback, height=500)

    # Download option
    st.download_button(
        label="📥 Download Feedback as .txt",
        data=review_feedback,
        file_name="resume_review_feedback.txt",
        mime="text/plain"
    )

elif uploaded_file and not job_description:
    st.warning("Please paste the job description to begin analysis.")
elif job_description and not uploaded_file:
    st.warning("Please upload a resume to begin analysis.")
