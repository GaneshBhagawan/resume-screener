import streamlit as st
import pdfplumber
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import anthropic

# Extract all text from uploaded PDF
def extract_text_from_pdf(pdf_file):
    text = ""
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted + " "
    return text.strip()

# Calculate match score using TF-IDF
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

# Generate improved resume using Claude AI
def generate_improved_resume(resume_text, jd_text, missing_keywords):
    client = anthropic.Anthropic(api_key=st.secrets["ANTHROPIC_API_KEY"])

    prompt = f"""You are a professional resume writer. Rewrite the candidate's resume to fit exactly ONE full A4 page.

CANDIDATE'S CURRENT RESUME:
{resume_text}

JOB DESCRIPTION:
{jd_text}

MISSING KEYWORDS TO INCLUDE:
{', '.join(missing_keywords)}

STRICT FORMATTING RULES — follow exactly:

1. Use ONLY real information from the candidate's resume. Do not invent anything.
2. Naturally include the missing keywords wherever they genuinely fit.
3. Return plain text only — no markdown, no asterisks (*), no hashtags (#), no bold symbols.
4. Use a hyphen (-) for all bullet points.

SECTION FORMAT RULES:

CONTACT INFORMATION
Write name on first line. Then email, phone, LinkedIn, GitHub each on a new line.

PROFESSIONAL SUMMARY
Write 3 to 4 full sentences describing the candidate's background, key skills, and career goal as it relates to the job description.

TECHNICAL SKILLS
List skills as short bullet points grouped by category. Each bullet point is one line. Example:
- Programming Languages: Python, Java, C++
- Web Technologies: HTML, CSS, JavaScript, React
- Tools & Platforms: Git, GitHub, VS Code, Streamlit
- Databases: MySQL, Firebase

PROJECTS
For each project write the project name on one line followed by 3 bullet points:
- What the project does in one clear sentence
- Technologies and tools used
- Outcome or result of the project
Leave one blank line between each project.

EDUCATION
Write each level separately with a blank line between them. Format:

B.Tech - Computer Science and Engineering
College name, City
Year of joining - Expected year of graduation
CGPA: X.X / 10

Intermediate (Class 12)
School name, City
Year of passing
Percentage: XX%

Secondary School Certificate (Class 10)
School name, City
Year of passing
Percentage: XX%

ACHIEVEMENTS & CERTIFICATIONS
Write achievements and certifications as separate bullet points. One point per line. Example:
- Achieved top 5 finalist position in college hackathon 2024
- Completed Python for Everybody course on Coursera

LANGUAGES KNOWN
Write as bullet points:
- Telugu (Native)
- English (Professional Proficiency)
- Add any other language found in the resume

CONCLUSION
Write 2 to 3 full sentences. Mention the specific job role from the job description. Express genuine interest in contributing to the company. Mention one or two key skills that make the candidate a strong fit for this particular role.

IMPORTANT:
- The total content must fill one complete A4 page — write enough in each section to fill the page properly
- Professional Summary, Conclusion must be in full sentence paragraph format
- Technical Skills, Projects, Achievements, Languages must be in bullet point format
- Education must be in the structured format shown above
- Do not add any section that is not listed above
- Do not use any symbols except hyphen (-) for bullets

Write the full resume now:"""

    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=2000,
        messages=[{"role": "user", "content": prompt}]
    )
    return message.content[0].text

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
    "to see how well they match — and get a professionally rewritten resume instantly."
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
        with st.spinner("Analysing your resume..."):
            resume_text = extract_text_from_pdf(uploaded_file)

            if not resume_text:
                st.error("Could not read text from the PDF. Try a different file.")
                st.stop()

            score = calculate_match_score(resume_text, jd_text)
            missing_keywords = find_missing_keywords(resume_text, jd_text)

        # ── Results ──
        st.subheader("Results")
        m1, m2, m3 = st.columns(3)
        m1.metric("Match Score", f"{score}%")
        m2.metric("Resume Words", len(resume_text.split()))
        m3.metric("Missing Keywords", len(missing_keywords))

        st.progress(int(score) / 100)

        if score >= 90:
            st.success("Excellent match! Your resume is very well aligned with this job.")
        elif score >= 70:
            st.success("Strong match! This resume fits the job description well.")
        elif score >= 45:
            st.warning("Moderate match. Adding the missing keywords below can improve it.")
        else:
            st.error("Low match. The resume needs significant updates for this role.")

        st.divider()

        # ── Missing Keywords ──
        st.subheader("Missing Keywords")
        st.markdown("These words appear in the JD but are missing from your resume:")

        if missing_keywords:
            cols = st.columns(5)
            for i, word in enumerate(missing_keywords):
                cols[i % 5].markdown(f"`{word}`")
        else:
            st.success("No major keywords missing!")

        st.divider()

        # ── AI Resume Generator (only if score < 90) ──
        if score < 90:
            st.subheader("✨ AI-Generated Improved Resume")
            st.markdown(
                f"Your match score is **{score}%** — below 90%. "
                "Here is a professionally rewritten version of your resume tailored to this job:"
            )

            with st.spinner("Generating your improved resume... please wait 20-30 seconds..."):
                try:
                    improved_resume = generate_improved_resume(
                        resume_text, jd_text, missing_keywords
                    )

                    # ── A4 styled display ──
                    st.markdown("""
                    <style>
                    .a4-resume {
                        background: #ffffff;
                        color: #1a1a1a;
                        font-family: 'Georgia', serif;
                        font-size: 13.5px;
                        line-height: 1.85;
                        padding: 52px 60px;
                        max-width: 794px;
                        min-height: 1123px;
                        margin: 0 auto;
                        border: 1px solid #d0d0d0;
                        border-radius: 4px;
                        white-space: pre-wrap;
                        word-wrap: break-word;
                        box-shadow: 0 2px 12px rgba(0,0,0,0.08);
                    }
                    </style>
                    """, unsafe_allow_html=True)

                    st.markdown(
                        f'<div class="a4-resume">{improved_resume}</div>',
                        unsafe_allow_html=True
                    )

                    st.divider()

                    # ── Copy box ──
                    st.text_area(
                        "Copy the resume text from here:",
                        value=improved_resume,
                        height=150,
                        help="Select all text (Ctrl+A) and copy (Ctrl+C)"
                    )

                    st.info(
                        "How to use this resume: Copy the text above → "
                        "Open Microsoft Word or Google Docs → "
                        "Paste it → Apply your preferred formatting → Save as PDF → Apply!"
                    )

                except Exception as e:
                    st.error(f"Could not generate resume. Error: {str(e)}")