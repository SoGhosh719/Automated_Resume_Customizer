import streamlit as st
from resume_parser import extract_resume_text
from utils import generate_custom_resume, compute_match_score

st.title("📄 AI Resume Tailoring Assistant")

uploaded_file = st.file_uploader("Upload your resume (.pdf or .docx)", type=["pdf", "docx"])
job_description = st.text_area("Paste the job description here")

if uploaded_file and job_description:
    with st.spinner("Reading your resume..."):
        resume_text = extract_resume_text(uploaded_file)

    with st.spinner("Tailoring your resume..."):
        tailored_resume = generate_custom_resume(resume_text, job_description)
        match_score = compute_match_score(resume_text, job_description)

    st.success("✅ Tailored Resume Ready!")
    st.subheader("🎯 Resume-Job Match Score")
    st.metric("Match Score", f"{match_score:.2f}%")
    
    st.subheader("📝 Tailored Resume")
    st.text_area("Output", value=tailored_resume, height=400)
