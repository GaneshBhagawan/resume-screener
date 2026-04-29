import streamlit as st
import pdfplumber
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Extract all text from uploaded PDF
def extract_text_from_pdf(pdf_file):
    text = ""
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted + " "
    return text.strip()

# Calculate match score using TF-IDF (lightweight, no heavy model needed)
def calculate_match_score(resume_text, jd_text):
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([resume_text, jd_text])
    score = cosine_similarity(vectors[0], vectors[1])
    return round(float(score[0][0]) * 100, 1)

# Find keywords in JD that are missing from resume
def find_missing_keywords(resume_text, jd_text):
    common_stopwords = {
        "with", "that", "this", "from", "have", "will", "your",
        "they", "their", "about", "should", "would", "which",
        "work", "also", "more", "must", "able", "good", "well"
    }
    jd_words = set(re.findall(r"\b[a-zA-Z]{4,}\b", jd_text.lower()))
    resume_words = set(re.findall(r"\b[a-zA-Z]{4,}\b", resume_text.lower()))
    missing = jd_words - resume_words - common_stopwords
    return sorted(list(missing))[:15]

# ─────────────────────────────────────────
#  Streamlit UI
# ─────────────────────────────────────────

st.set_page_config(
    page_title="AI Resume Screener",
    page_icon="📄",
    layout="wide"
)

st.title("📄 AI Resume Screener")
st.markdown(
    "Paste a **Job Description** and upload a **Resume PDF** "
    "to see how well they match."
)

st.divider()

col1, col2 = st.columns(2)

with col1:
    st.subheader("Job Description")
    jd_text = st.text_area(
        "Paste the full job description here",
        height=300,
        placeholder="e.g. We are looking for a Python developer with..."
    )

with col2:
    st.subheader("Resume")
    uploaded_file = st.file_uploader(
        "Upload resume as PDF",
        type=["pdf"]
    )
    if uploaded_file:
        st.success(f"Uploaded: {uploaded_file.name}")

st.divider()

if st.button("Analyse Match", use_container_width=True, type="primary"):
    if not jd_text.strip():
        st.warning("Please paste a job description first.")
    elif uploaded_file is None:
        st.warning("Please upload a resume PDF.")
    else:
        with st.spinner("Analysing..."):
            resume_text = extract_text_from_pdf(uploaded_file)

            if not resume_text:
                st.error("Could not read text from the PDF. Try a different file.")
                st.stop()

            score = calculate_match_score(resume_text, jd_text)
            missing_keywords = find_missing_keywords(resume_text, jd_text)

        st.subheader("Results")

        m1, m2, m3 = st.columns(3)
        m1.metric("Match Score", f"{score}%")
        m2.metric("Resume Words", len(resume_text.split()))
        m3.metric("Missing Keywords", len(missing_keywords))

        st.progress(int(score) / 100)

        if score >= 70:
            st.success("Strong match! This resume fits the job description well.")
        elif score >= 45:
            st.warning("Moderate match. Adding the missing keywords below can improve it.")
        else:
            st.error("Low match. The resume needs significant updates for this role.")

        st.divider()
        st.subheader("Missing Keywords")
        st.markdown("These words appear in the JD but are missing from the resume:")

        if missing_keywords:
            cols = st.columns(5)
            for i, word in enumerate(missing_keywords):
                cols[i % 5].markdown(f"`{word}`")
        else:
            st.success("No major keywords missing!")