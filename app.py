import streamlit as st
import pdfplumber
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from groq import Groq
from fpdf import FPDF
import tempfile
import os

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

# Generate improved resume using Groq
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
4. The total resume length must fill 1 to 1.5 A4 pages when printed — enough content to look complete but not overcrowded.

SECTION BY SECTION FORMAT:

--- SECTION 1: CONTACT INFORMATION ---
Write the candidate's name on the first line in CAPITALS.
Then write each of the following on a separate line: email, phone number, LinkedIn URL, GitHub URL (only if present in resume).
No label needed for this section — just the raw contact details.

--- SECTION 2: PROFESSIONAL SUMMARY ---
Section heading: PROFESSIONAL SUMMARY
Write exactly 3 clear, professional sentences in paragraph form.
Sentence 1: Who the candidate is and their academic background.
Sentence 2: Their key technical skills and how they relate to this specific job role.
Sentence 3: Their career goal and availability for this specific internship/job role mentioned in the JD.
Keep language simple, honest, and confident. Do not use fancy or complex words.

--- SECTION 3: TECHNICAL SKILLS ---
Section heading: TECHNICAL SKILLS
List skills as clean bullet points grouped by category. Use this format exactly:
- Programming Languages: [list them]
- Web Technologies: [list them]
- Tools and Platforms: [list them]
- Databases: [list them]
- Operating Systems: [only if relevant]
Only include skill categories that exist in the candidate's resume. Maximum 6 bullet points total.

--- SECTION 4: PROJECTS ---
Section heading: PROJECTS
For each project from the candidate's resume, write:
Line 1: Project name only (no bullet, just the name)
Line 2: - One sentence clearly explaining what the project does and its purpose.
Line 3: - One sentence listing the technologies used and the result or outcome.
Leave exactly one blank line between projects.
Maximum 3 projects. Only 2 bullet points per project. Keep sentences short and clear.

--- SECTION 5: EDUCATION ---
Section heading: EDUCATION
Write each level as a separate block with one blank line between them.
Format for each block:
Degree name
College/School name, City
Year range or year of passing
CGPA: X.X / 10 OR Percentage: XX%

Write exactly 3 blocks in this order: B.Tech first, then Intermediate (Class 12), then Secondary School Certificate (Class 10).
If any detail is missing from the resume, write "Details not provided" for that line.

--- SECTION 6: ACHIEVEMENTS AND CERTIFICATIONS ---
Section heading: ACHIEVEMENTS AND CERTIFICATIONS
Write each achievement and each certification as a separate bullet point.
Format: - [Achievement or certification description in one clear sentence]
Keep each point short — maximum 15 words per bullet.
Maximum 5 bullet points total.

--- SECTION 7: LANGUAGES KNOWN ---
Section heading: LANGUAGES KNOWN
Write each language as a bullet point:
- [Language] ([Proficiency level])
Include Telugu and English always. Add any other language found in the resume.

--- SECTION 8: CONCLUSION ---
Section heading: CONCLUSION
Write exactly 2 to 3 sentences.
Sentence 1: Express genuine enthusiasm for the specific job role mentioned in the JD and name the company if available.
Sentence 2: Mention 1 or 2 specific skills from the candidate's resume that directly match the JD requirements.
Sentence 3: Express commitment and readiness to contribute.
Keep it professional, warm, and specific to this job. Do not use generic phrases.

SPACING RULES:
- Leave exactly one blank line between each section heading and its content.
- Leave exactly one blank line between sections.
- Do not add extra blank lines inside sections.

Write the complete resume now:"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=2500,
        temperature=0.5
    )
    return response.choices[0].message.content

# Generate PDF from resume text
def generate_pdf(resume_text):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_margins(20, 20, 20)
    pdf.set_auto_page_break(auto=True, margin=20)

    lines = resume_text.split('\n')

    section_headings = [
        "PROFESSIONAL SUMMARY", "TECHNICAL SKILLS", "PROJECTS",
        "EDUCATION", "ACHIEVEMENTS AND CERTIFICATIONS",
        "LANGUAGES KNOWN", "CONCLUSION", "CONTACT INFORMATION"
    ]

    for line in lines:
        stripped = line.strip()

        if not stripped:
            pdf.ln(3)
            continue

        # Detect if it's the name (first non-empty line = all caps and short)
        if stripped.isupper() and len(stripped.split()) <= 4 and stripped not in section_headings:
            pdf.set_font("Helvetica", "B", 14)
            pdf.set_text_color(0, 0, 0)
            pdf.cell(0, 8, stripped, ln=True, align='C')
            pdf.ln(1)

        # Section headings
        elif stripped in section_headings:
            pdf.ln(2)
            pdf.set_font("Helvetica", "B", 10)
            pdf.set_text_color(0, 0, 0)
            pdf.cell(0, 6, stripped, ln=True)
            pdf.set_draw_color(0, 0, 0)
            pdf.set_line_width(0.3)
            pdf.line(20, pdf.get_y(), 190, pdf.get_y())
            pdf.ln(2)

        # Bullet points
        elif stripped.startswith("- "):
            pdf.set_font("Helvetica", "", 9)
            pdf.set_text_color(40, 40, 40)
            pdf.set_x(24)
            pdf.multi_cell(0, 5.5, stripped, align='L')

        # Contact lines and other content
        else:
            pdf.set_font("Helvetica", "", 9)
            pdf.set_text_color(40, 40, 40)
            pdf.multi_cell(0, 5.5, stripped, align='L')

    # Save to temp file
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
                "Here is a professionally rewritten ATS-optimized version of your resume:"
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

                    # ── PDF Download ──
                    st.subheader("⬇ Download Generated Resume as PDF")
                    st.markdown("Click the button below to download your improved resume as a ready-to-send PDF file.")

                    with st.spinner("Preparing your PDF..."):
                        pdf_path = generate_pdf(improved_resume)
                        with open(pdf_path, "rb") as f:
                            pdf_bytes = f.read()
                        os.unlink(pdf_path)

                    st.download_button(
                        label="Download Resume as PDF",
                        data=pdf_bytes,
                        file_name="improved_resume.pdf",
                        mime="application/pdf",
                        use_container_width=True
                    )

                    st.divider()

                    # ── Copy box ──
                    st.subheader("Copy Resume Text")
                    st.text_area(
                        "Select all and copy (Ctrl+A then Ctrl+C):",
                        value=improved_resume,
                        height=200,
                        help="Select all text (Ctrl+A) and copy (Ctrl+C)"
                    )

                    st.info(
                        "How to use: Download the PDF above and send directly — "
                        "or copy the text, paste into Word/Google Docs, format, and save as PDF."
                    )

                except Exception as e:
                    st.error(f"Could not generate resume. Error: {str(e)}")