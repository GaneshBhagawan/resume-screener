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

def clean_text_for_pdf(text):
    replacements = {
        "\u2018": "'", "\u2019": "'",
        "\u201c": '"', "\u201d": '"',
        "\u2013": "-", "\u2014": "-",
        "\u2022": "-", "\u2026": "...",
        "\u00e9": "e", "\u00e8": "e",
        "\u00e0": "a", "\u00e2": "a",
        "\u00f4": "o", "\u00fb": "u",
        "\u00ee": "i", "\u00e7": "c",
        "\u2010": "-", "\u2011": "-",
        "\u00b7": "-", "\u00a0": " ",
        "\u00bc": "1/4", "\u00bd": "1/2",
        "\u00be": "3/4", "\u00d7": "x",
        "\u00f7": "/",
    }
    for unicode_char, ascii_char in replacements.items():
        text = text.replace(unicode_char, ascii_char)
    result = ""
    for char in text:
        if ord(char) < 128:
            result += char
        else:
            result += " "
    return result

def generate_improved_resume(resume_text, jd_text, missing_keywords):
    client = Groq(api_key=st.secrets["GROQ_API_KEY"])

    prompt = f"""You are an expert ATS-optimized resume writer. Your goal is to rewrite the candidate's resume so it scores 95% or above when matched against the given job description using ATS systems.

CANDIDATE RESUME:
{resume_text}

JOB DESCRIPTION:
{jd_text}

MISSING KEYWORDS TO INCLUDE NATURALLY:
{', '.join(missing_keywords)}

OUTPUT RULES:
1. Use ONLY real details from the candidate's resume. Never invent anything.
2. Naturally include missing keywords wherever they genuinely fit.
3. Return plain text only. No markdown. No asterisks. No hashtags. No bold symbols.
4. Use only basic ASCII characters. Use straight apostrophes and quotes only. Use hyphen (-) for bullets.
5. The total resume must fill 1 to 1.5 A4 pages.

SECTION FORMAT:

CONTACT INFORMATION
Candidate name on first line in CAPITALS.
Then email, phone, LinkedIn, GitHub each on a separate line.

PROFESSIONAL SUMMARY
Write exactly 3 simple clear sentences.
Sentence 1: Academic background and who the candidate is.
Sentence 2: Key technical skills relevant to the job role.
Sentence 3: Career goal and availability for this specific role.

TECHNICAL SKILLS
- Programming Languages: [list]
- Web Technologies: [list]
- Tools and Platforms: [list]
- Databases: [list]
Maximum 6 bullet points. Only include what exists in the resume.

PROJECTS
Project name on one line (no bullet).
- One sentence: what the project does and its purpose.
- One sentence: technologies used and outcome.
One blank line between projects. Maximum 3 projects.

EDUCATION
B.Tech - [Branch]
[College], [City]
[Year] - [Year]
CGPA: [value] / 10

Intermediate (Class 12)
[School], [City]
[Year]
Percentage: [value]%

Secondary School Certificate (Class 10)
[School], [City]
[Year]
Percentage: [value]%

ACHIEVEMENTS AND CERTIFICATIONS
- [One sentence, max 15 words]
Maximum 5 bullet points.

LANGUAGES KNOWN
- Telugu (Native)
- English (Professional Proficiency)
Add others from resume.

CONCLUSION
2 to 3 sentences. Mention the specific job role and company. Mention 1 or 2 matching skills. Express commitment.

Write the complete resume now. Use only basic English characters. No special symbols."""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=2500,
        temperature=0.5
    )
    return response.choices[0].message.content

def generate_pdf(resume_text):
    cleaned_text = clean_text_for_pdf(resume_text)

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

    lines = cleaned_text.split('\n')
    first_nonblank_done = False

    for line in lines:
        stripped = line.strip()

        if not stripped:
            pdf.ln(2)
            continue

        if not first_nonblank_done:
            first_nonblank_done = True
            pdf.set_font("Helvetica", "B", 13)
            pdf.set_text_color(0, 0, 0)
            pdf.cell(0, 7, stripped, ln=True, align='C')
            pdf.ln(1)
            continue

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

        if stripped.startswith("- "):
            pdf.set_font("Helvetica", "", 9)
            pdf.set_text_color(40, 40, 40)
            content = stripped[2:]
            pdf.set_x(19)
            pdf.cell(4, 5, "-", ln=False)
            pdf.set_x(23)
            pdf.multi_cell(172, 5, content, align='L')
            continue

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