from huggingface_hub import InferenceClient
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import streamlit as st
import traceback

# Initialize Hugging Face client
client = InferenceClient(
    model="HuggingFaceH4/zephyr-7b-beta",
    token=st.secrets["HUGGINGFACE_TOKEN"]
)

# Clean input text
def clean_text(text):
    return re.sub(r'\s+', ' ', text.strip()) if text else ""

# 🔍 Expert-level resume-job fit review function
def review_resume_for_job_fit(resume_text, job_description):
    system_prompt = (
        "You are an executive career strategist and resume reviewer working with top consulting firms and tech companies. "
        "You specialize in analyzing resumes against job descriptions and offering detailed, structured, and actionable feedback "
        "that improves alignment, clarity, and impact. You DO NOT rewrite resumes. You provide a feedback report using a consulting-level format."
    )

    user_prompt = f"""
You are given a resume and a job description. Your task is to critically evaluate the resume in relation to the job description.

Return the response using the following sections and formatting:

---

### ✅ 1. OVERALL FIT ASSESSMENT
Summarize if the resume is a strong, moderate, or weak match. Mention the tone, relevance, and completeness.

---

### 🔍 2. STRENGTHS MAPPED TO ROLE
Present a markdown table:

| **Job Requirement** | **Resume Evidence** |
|---------------------|---------------------|
| Requirement 1       | Evidence or phrase from resume |
| Requirement 2       | ... |

Only include 5–7 key points. Be concise and insightful.

---

### ❗ 3. GAPS OR UNDEREMPHASIZED AREAS
List job qualifications or expectations that are not clearly addressed in the resume. Focus on what's missing or weakly expressed.

- Example: No mention of stakeholder communication or reporting
- Example: Lacks cloud security tools (e.g., IAM, encryption, authentication)

---

### 🛠 4. STRATEGIC REFRAMING SUGGESTIONS
Advise how to reposition or rewrite existing experiences to better align with the role. Recommend ways to showcase transferable skills or adjacent expertise.

- Reframe academic mentoring as project leadership
- Reposition digital marketing simulations as data-driven performance tracking

---

### 💡 5. HIGHLIGHTS FOR COVER LETTER / INTERVIEW
Tell the user what to emphasize to stand out.

- Emphasize your multi-domain background across chemistry, analytics, and AI
- Showcase your use of cloud/AI tools with secure data handling

---

Resume:
\"\"\"{resume_text}\"\"\"

Job Description:
\"\"\"{job_description}\"\"\"
"""

    try:
        response = client.chat_completion(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=1200,
            temperature=0.25
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        st.error("⚠️ Error calling Hugging Face API.")
        st.text(traceback.format_exc())
        return f"⚠️ Error: {str(e)}"

# 🔢 Text similarity match score
def compute_match_score(resume_text, job_description):
    try:
        resume_text = clean_text(resume_text)
        job_description = clean_text(job_description)
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform([resume_text, job_description])
        score = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        return round(score * 100, 2)
    except Exception as e:
        st.warning(f"⚠️ Match score calculation failed: {str(e)}")
        return 0.0
