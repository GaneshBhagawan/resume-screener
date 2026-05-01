import streamlit as st
import pdfplumber
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from groq import Groq
from fpdf import FPDF
import tempfile
import os

def extract_text_from_pdf(pdf_file):
    text = ""
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted + " "
    return text.strip()

def calculate_match_score(resume_text, jd_text):
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([resume_text, jd_text])
    score = cosine_similarity(vectors[0], vectors[1])
    return round(float(score[0][0]) * 100, 1)

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

def generate_improved_resume(resume_text, jd_text, missing_keywords):
    client = Groq(api_key=st.secrets["GROQ_API_KEY"])

    prompt = f"""You are an expert ATS-optimized resume writer. Your goal is to rewrite the candidate's resume so it scores 95% or above when matched against the given job description using ATS systems.

CANDIDATE RESUME:
{resume_text}

JOB DESCRIPTION:
{jd_text}

MISSING KEYWORDS TO INCLUDE NATURALLY:
{', '.join(missing_keywords)}

OUTPUT RULES — follow every rule strictly:

1. Use ONLY real details from the candidate's resume. Never invent fake experience, fake companies, or fake projects.
2. Naturally weave in as many missing keywords as possible wherever they genuinely fit.
3. Return plain text only. No markdown. No asterisks. No hashtags. No bold symbols. No special characters except hyphen (-) for bullets.
4. The total resume length must fill 1 to 1.5 A4 pages when printed.

SECTION BY SECTION FORMAT:

CONTACT INFORMATION
Write the candidate's name on the first line in CAPITALS.
Then write each of the following on a separate line: email, phone number, LinkedIn URL, GitHub URL (only if present in resume).

PROFESSIONAL SUMMARY
Section heading: PROFESSIONAL SUMMARY
Write exactly 3 clear professional sentences in paragraph form.
Sentence 1: Who the candidate is and their academic background.
Sentence 2: Their key technical skills and how they relate to this specific job role.
Sentence 3: Their career goal and availability for this specific internship or job role mentioned in the JD.
Keep language simple, honest, and confident.

TECHNICAL SKILLS
Section heading: TECHNICAL SKILLS
List skills as clean bullet points grouped by category:
- Programming Languages: [list them]
- Web Technologies: [list them]
- Tools and Platforms: [list them]
- Databases: [list them]
- Operating Systems: [only if relevant]
Only include categories that exist in the candidate's resume. Maximum 6 bullet points.

PROJECTS
Section heading: PROJECTS
For each project write:
Line 1: Project name only (no bullet)
Line 2: - One sentence explaining what the project does and its purpose.
Line 3: - One sentence listing technologies used and the outcome.
Leave one blank line between projects. Maximum 3 projects. Only 2 bullet points per project.

EDUCATION
Section heading: EDUCATION
Write 3 blocks in this order with one blank line between each:

B.Tech - [Branch name]
[College name], [City]
[Year] - [Year]
CGPA: [value] / 10

Intermediate (Class 12)
[School name], [City]
[Year]
Percentage: [value]%

Secondary School Certificate (Class 10)
[School name], [City]
[Year]
Percentage: [value]%

ACHIEVEMENTS AND CERTIFICATIONS
Section heading: ACHIEVEMENTS AND CERTIFICATIONS
Write each as a separate bullet point:
- [One clear sentence, maximum 15 words]
Maximum 5 bullet points.

LANGUAGES KNOWN
Section heading: LANGUAGES KNOWN
- [Language] ([Proficiency])
Always include Telugu and English. Add others from resume.

CONCLUSION
Section heading: CONCLUSION
Write exactly 2 to 3 sentences.
Sentence 1: Enthusiasm for the specific job role and company name if available.
Sentence 2: One or two specific skills that match the JD.
Sentence 3: Commitment and readiness to contribute.

SPACING RULES:
- One blank line between each section heading and its content.
- One blank line between sections.
- No extra blank lines inside sections.

Write the complete resume now:"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=2500,
        temperature=0.5
    )
    return response.choices[0].message.content

def generate_pdf(resume_text):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_margins(15, 15, 15)
    pdf.set_auto_page_break(auto=True, margin=15)

    section_headings = [
        "PROFESSIONAL SUMMARY", "TECHNICAL SKILLS", "PROJECTS",
        "EDUCATION", "ACHIEVEMENTS AND CERTIFICATIONS",
        "ACHIEVEMENTS & CERTIFICATIONS", "LANGUAGES KNOWN",
        "CONCLUSION", "CONTACT INFORMATION"
    ]

    lines = resume_text.split('\n')
    first_nonblank_done = False

    for line in lines:
        stripped = line.strip()

        if not stripped:
            pdf.ln(2)
            continue

        # First non-empty line = candidate name
        if not first_nonblank_done:
            first_nonblank_done = True
            pdf.set_font("Helvetica", "B", 13)
            pdf.set_text_color(0, 0, 0)
            pdf.cell(0, 7, stripped, ln=True, align='C')
            pdf.ln(1)
            continue

        # Section headings
        if stripped.upper() in [s.upper() for s in section_headings]:
            pdf.ln(3)
            pdf.set_font("Helvetica", "B", 9)
            pdf.set_text_color(0, 0, 0)
            pdf.cell(0, 5, stripped.upper(), ln=True)
            pdf.set_draw_color(0, 0, 0)
            pdf.set_line_width(0.3)
            pdf.line(15, pdf.get_y(), 195, pdf.get_y())
            pdf.ln(2)
            continue

        # Bullet points
        if stripped.startswith("- "):
            pdf.set_font("Helvetica", "", 9)
            pdf.set_text_color(40, 40, 40)
            content = stripped[2:]
            pdf.set_x(19)
            pdf.cell(4, 5, "-", ln=False)
            pdf.set_x(23)
            pdf.multi_cell(172, 5, content, align='L')
            continue

        # Regular text
        pdf.set_font("Helvetica", "", 9)
        pdf.set_text_color(40, 40, 40)
        pdf.multi_cell(180, 5, stripped, align='L')

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    pdf.output(tmp.name)
    return tmp.name

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
    "to see how well they match — and get a professionally rewritten ATS-optimized resume instantly."
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

        st.subheader("Missing Keywords")
        st.markdown("These words appear in the JD but are missing from your resume:")
        if missing_keywords:
            cols = st.columns(5)
            for i, word in enumerate(missing_keywords):
                cols[i % 5].markdown(f"`{word}`")
        else:
            st.success("No major keywords missing!")

        st.divider()

        if score < 90:
            st.subheader("✨ AI-Generated Improved Resume")
            st.markdown(
                f"Your match score is **{score}%** — below 90%. "
                "Here is a professionally rewritten ATS-optimized version of your resume:"
            )

            with st.spinner("Generating your improved resume... please wait 20-30 seconds..."):
                try:
                    improved_resume = generate_improved_resume(
                        resume_text, jd_text, missing_keywords
                    )

                    # A4 display
                    st.markdown("""
                    <style>
                    .a4-resume {
                        background: #ffffff;
                        color: #1a1a1a;
                        font-family: 'Arial', sans-serif;
                        font-size: 13px;
                        line-height: 1.7;
                        padding: 50px 60px;
                        max-width: 794px;
                        min-height: 1000px;
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

                    # PDF Download
                    st.subheader("⬇ Download Generated Resume as PDF")
                    st.markdown("Click the button below to download your improved resume as a ready-to-send PDF.")

                    with st.spinner("Preparing your PDF..."):
                        pdf_path = generate_pdf(improved_resume)
                        with open(pdf_path, "rb") as f:
                            pdf_bytes = f.read()
                        os.unlink(pdf_path)

                    st.download_button(
                        label="⬇ Download Resume as PDF",
                        data=pdf_bytes,
                        file_name="improved_resume.pdf",
                        mime="application/pdf",
                        use_container_width=True
                    )

                    st.divider()

                    # Copy box
                    st.subheader("Copy Resume Text")
                    st.text_area(
                        "Select all and copy (Ctrl+A then Ctrl+C):",
                        value=improved_resume,
                        height=200
                    )

                    st.info(
                        "How to use: Download the PDF above and send directly — "
                        "or copy the text, paste into Word or Google Docs, format, and save as PDF."
                    )

                except Exception as e:
                    st.error(f"Could not generate resume. Error: {str(e)}")