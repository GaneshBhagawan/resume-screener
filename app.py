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
    }
    for k, v in replacements.items():
        text = text.replace(k, v)
    result = ""
    for char in text:
        if ord(char) < 128:
            result += char
        else:
            result += " "
    return result

def generate_improved_resume(resume_text, jd_text, missing_keywords):
    client = Groq(api_key=st.secrets["GROQ_API_KEY"])

    prompt = f"""You are an expert ATS-optimized resume writer. Rewrite the candidate's resume to score 95% or above against the job description.

CANDIDATE RESUME:
{resume_text}

JOB DESCRIPTION:
{jd_text}

MISSING KEYWORDS TO INCLUDE:
{', '.join(missing_keywords)}

CRITICAL RULES:
1. Use ONLY real details from the candidate's resume. Never invent anything fake.
2. Include missing keywords naturally wherever they fit.
3. Use ONLY basic ASCII characters. No special quotes, no em dashes, no bullet symbols. Use hyphen (-) only.
4. No markdown, no asterisks, no hashtags.
5. Total length: 1 to 1.5 A4 pages.

OUTPUT FORMAT — follow this structure exactly with these exact section markers:

##NAME##
[Candidate full name in CAPITALS]

##EMAIL##
[email address]

##PHONE##
[phone number]

##LINKEDIN##
[LinkedIn URL or blank]

##GITHUB##
[GitHub URL or blank]

##SUMMARY##
[Write exactly 3 sentences. Sentence 1: who the candidate is and academic background. Sentence 2: key technical skills relevant to this job. Sentence 3: availability and enthusiasm for this specific role and company.]

##SKILLS##
[List skills as bullet points grouped by category. Use format:
- Category Name: skill1, skill2, skill3
Maximum 6 lines. Only include what exists in the resume.]

##PROJECTS##
[For each project:
PROJECT: [project name]
- [One sentence: what it does and its purpose]
- [One sentence: technologies used and outcome]
Leave one blank line between projects. Maximum 3 projects.]

##EDUCATION##
[Write exactly 3 blocks:

BTECH: [Degree name] | [College], [City] | [Year]-[Year] | CGPA: [value]/10

CLASS12: [Stream if known] | [School name], [City] | [Year] | Percentage: [value]%

CLASS10: Secondary School Certificate | [School name], [City] | [Year] | Percentage: [value]%

If any detail is missing from the resume write: Not provided]

##ACHIEVEMENTS##
[Each as a bullet point:
- [One sentence achievement or certification, max 15 words]
Maximum 5 points.]

##LANGUAGES##
[Bullet points:
- Telugu (Native)
- English (Professional Proficiency)
Add others from resume.]

##CONCLUSION##
[2-3 sentences. Sentence 1: enthusiasm for the specific role and company. Sentence 2: specific matching skills. Sentence 3: commitment to contribute.]

Write the complete resume now using these exact section markers:"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=2500,
        temperature=0.5
    )
    return response.choices[0].message.content

def parse_resume_sections(raw_text):
    sections = {}
    markers = [
        "##NAME##", "##EMAIL##", "##PHONE##", "##LINKEDIN##", "##GITHUB##",
        "##SUMMARY##", "##SKILLS##", "##PROJECTS##", "##EDUCATION##",
        "##ACHIEVEMENTS##", "##LANGUAGES##", "##CONCLUSION##"
    ]
    for i, marker in enumerate(markers):
        if marker in raw_text:
            start = raw_text.index(marker) + len(marker)
            end = len(raw_text)
            for next_marker in markers[i+1:]:
                if next_marker in raw_text:
                    end = raw_text.index(next_marker)
                    break
            sections[marker] = raw_text[start:end].strip()
    return sections

def generate_pdf(raw_resume_text):
    sections = parse_resume_sections(raw_resume_text)

    pdf = FPDF()
    pdf.add_page()
    pdf.set_margins(18, 18, 18)
    pdf.set_auto_page_break(auto=True, margin=15)

    effective_width = 174

    def add_section_heading(title):
        pdf.ln(4)
        pdf.set_font("Helvetica", "B", 10)
        pdf.set_text_color(0, 0, 0)
        pdf.cell(0, 6, title.upper(), ln=True)
        pdf.set_draw_color(0, 0, 0)
        pdf.set_line_width(0.4)
        pdf.line(18, pdf.get_y(), 192, pdf.get_y())
        pdf.ln(3)

    def add_body_text(text, indent=0):
        pdf.set_font("Helvetica", "", 9)
        pdf.set_text_color(50, 50, 50)
        pdf.set_x(18 + indent)
        pdf.multi_cell(effective_width - indent, 5, clean_text_for_pdf(text), align='L')

    def add_bullet(text):
        pdf.set_font("Helvetica", "", 9)
        pdf.set_text_color(50, 50, 50)
        cleaned = clean_text_for_pdf(text.lstrip("- ").strip())
        pdf.set_x(20)
        pdf.cell(5, 5, "-", ln=False)
        pdf.set_x(25)
        pdf.multi_cell(effective_width - 7, 5, cleaned, align='L')

    # ── Name (large, bold, centered) ──
    name = clean_text_for_pdf(sections.get("##NAME##", "").strip())
    if name:
        pdf.set_font("Helvetica", "B", 16)
        pdf.set_text_color(0, 0, 0)
        pdf.cell(0, 10, name, ln=True, align='C')
        pdf.ln(1)

    # ── Contact row (email left, phone right | linkedin left, github right) ──
    email = clean_text_for_pdf(sections.get("##EMAIL##", "").strip())
    phone = clean_text_for_pdf(sections.get("##PHONE##", "").strip())
    linkedin = clean_text_for_pdf(sections.get("##LINKEDIN##", "").strip())
    github = clean_text_for_pdf(sections.get("##GITHUB##", "").strip())

    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(50, 50, 50)
    half = effective_width / 2

    if email or phone:
        pdf.set_x(18)
        pdf.cell(half, 5, f"Email: {email}", align='L', ln=False)
        pdf.cell(half, 5, f"Phone: {phone}", align='R', ln=True)

    if linkedin or github:
        pdf.set_x(18)
        pdf.cell(half, 5, f"LinkedIn: {linkedin}" if linkedin else "", align='L', ln=False)
        pdf.cell(half, 5, f"GitHub: {github}" if github else "", align='R', ln=True)

    pdf.ln(2)
    pdf.set_draw_color(180, 180, 180)
    pdf.set_line_width(0.2)
    pdf.line(18, pdf.get_y(), 192, pdf.get_y())
    pdf.ln(3)

    # ── Professional Summary ──
    summary = sections.get("##SUMMARY##", "")
    if summary:
        add_section_heading("Professional Summary")
        add_body_text(summary)
        pdf.ln(2)

    # ── Technical Skills ──
    skills = sections.get("##SKILLS##", "")
    if skills:
        add_section_heading("Technical Skills")
        for line in skills.split('\n'):
            line = line.strip()
            if not line:
                continue
            if line.startswith("-"):
                parts = line.lstrip("- ").split(":", 1)
                if len(parts) == 2:
                    pdf.set_font("Helvetica", "B", 9)
                    pdf.set_text_color(0, 0, 0)
                    pdf.set_x(20)
                    label = clean_text_for_pdf(parts[0].strip()) + ":"
                    value = clean_text_for_pdf(parts[1].strip())
                    pdf.cell(45, 5, label, ln=False)
                    pdf.set_font("Helvetica", "", 9)
                    pdf.set_text_color(50, 50, 50)
                    pdf.multi_cell(effective_width - 47, 5, value, align='L')
                else:
                    add_bullet(line)
            else:
                add_body_text(line)
        pdf.ln(2)

    # ── Projects ──
    projects = sections.get("##PROJECTS##", "")
    if projects:
        add_section_heading("Projects")
        for line in projects.split('\n'):
            line = line.strip()
            if not line:
                pdf.ln(2)
                continue
            if line.upper().startswith("PROJECT:"):
                proj_name = clean_text_for_pdf(line.split(":", 1)[1].strip())
                pdf.set_font("Helvetica", "B", 9)
                pdf.set_text_color(0, 0, 0)
                pdf.set_x(18)
                pdf.cell(0, 5, proj_name, ln=True)
            elif line.startswith("-"):
                add_bullet(line)
            else:
                add_body_text(line)
        pdf.ln(2)

    # ── Education ──
    education = sections.get("##EDUCATION##", "")
    if education:
        add_section_heading("Education")
        for line in education.split('\n'):
            line = line.strip()
            if not line:
                pdf.ln(2)
                continue
            if line.upper().startswith("BTECH:"):
                parts = line.split(":", 1)[1].strip().split("|")
                pdf.set_font("Helvetica", "B", 9)
                pdf.set_text_color(0, 0, 0)
                pdf.set_x(18)
                degree = clean_text_for_pdf(parts[0].strip()) if len(parts) > 0 else ""
                college = clean_text_for_pdf(parts[1].strip()) if len(parts) > 1 else ""
                years = clean_text_for_pdf(parts[2].strip()) if len(parts) > 2 else ""
                cgpa = clean_text_for_pdf(parts[3].strip()) if len(parts) > 3 else ""
                pdf.cell(0, 5, degree, ln=True)
                pdf.set_font("Helvetica", "", 9)
                pdf.set_text_color(50, 50, 50)
                pdf.set_x(18)
                pdf.cell(effective_width / 2, 5, college, align='L', ln=False)
                pdf.cell(effective_width / 2, 5, years, align='R', ln=True)
                if cgpa:
                    pdf.set_x(18)
                    pdf.cell(0, 5, cgpa, ln=True)
            elif line.upper().startswith("CLASS12:"):
                parts = line.split(":", 1)[1].strip().split("|")
                pdf.set_font("Helvetica", "B", 9)
                pdf.set_text_color(0, 0, 0)
                pdf.set_x(18)
                pdf.cell(0, 5, "Intermediate (Class 12)", ln=True)
                pdf.set_font("Helvetica", "", 9)
                pdf.set_text_color(50, 50, 50)
                school = clean_text_for_pdf(parts[0].strip()) if len(parts) > 0 else ""
                year = clean_text_for_pdf(parts[1].strip()) if len(parts) > 1 else ""
                pct = clean_text_for_pdf(parts[2].strip()) if len(parts) > 2 else ""
                pdf.set_x(18)
                pdf.cell(effective_width / 2, 5, school, align='L', ln=False)
                pdf.cell(effective_width / 2, 5, year, align='R', ln=True)
                if pct:
                    pdf.set_x(18)
                    pdf.cell(0, 5, pct, ln=True)
            elif line.upper().startswith("CLASS10:"):
                parts = line.split(":", 1)[1].strip().split("|")
                pdf.set_font("Helvetica", "B", 9)
                pdf.set_text_color(0, 0, 0)
                pdf.set_x(18)
                pdf.cell(0, 5, "Secondary School Certificate (Class 10)", ln=True)
                pdf.set_font("Helvetica", "", 9)
                pdf.set_text_color(50, 50, 50)
                school = clean_text_for_pdf(parts[0].strip()) if len(parts) > 0 else ""
                year = clean_text_for_pdf(parts[1].strip()) if len(parts) > 1 else ""
                pct = clean_text_for_pdf(parts[2].strip()) if len(parts) > 2 else ""
                pdf.set_x(18)
                pdf.cell(effective_width / 2, 5, school, align='L', ln=False)
                pdf.cell(effective_width / 2, 5, year, align='R', ln=True)
                if pct:
                    pdf.set_x(18)
                    pdf.cell(0, 5, pct, ln=True)
            else:
                add_body_text(line)
        pdf.ln(2)

    # ── Achievements ──
    achievements = sections.get("##ACHIEVEMENTS##", "")
    if achievements:
        add_section_heading("Achievements and Certifications")
        for line in achievements.split('\n'):
            line = line.strip()
            if not line:
                continue
            if line.startswith("-"):
                add_bullet(line)
            else:
                add_body_text(line)
        pdf.ln(2)

    # ── Languages ──
    languages = sections.get("##LANGUAGES##", "")
    if languages:
        add_section_heading("Languages Known")
        for line in languages.split('\n'):
            line = line.strip()
            if not line:
                continue
            if line.startswith("-"):
                add_bullet(line)
            else:
                add_body_text(line)
        pdf.ln(2)

    # ── Conclusion ──
    conclusion = sections.get("##CONCLUSION##", "")
    if conclusion:
        add_section_heading("Conclusion")
        add_body_text(conclusion)
        pdf.ln(2)

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    pdf.output(tmp.name)
    return tmp.name

def build_display_html(raw_text):
    sections = parse_resume_sections(raw_text)

    name = sections.get("##NAME##", "").strip()
    email = sections.get("##EMAIL##", "").strip()
    phone = sections.get("##PHONE##", "").strip()
    linkedin = sections.get("##LINKEDIN##", "").strip()
    github = sections.get("##GITHUB##", "").strip()
    summary = sections.get("##SUMMARY##", "").strip()
    skills_raw = sections.get("##SKILLS##", "").strip()
    projects_raw = sections.get("##PROJECTS##", "").strip()
    education_raw = sections.get("##EDUCATION##", "").strip()
    achievements_raw = sections.get("##ACHIEVEMENTS##", "").strip()
    languages_raw = sections.get("##LANGUAGES##", "").strip()
    conclusion = sections.get("##CONCLUSION##", "").strip()

    def bullet_lines(raw):
        html = ""
        for line in raw.split('\n'):
            line = line.strip()
            if not line:
                continue
            if line.startswith("-"):
                html += f"<div class='bul'><span class='bd'>-</span> {line.lstrip('- ').strip()}</div>"
            else:
                html += f"<p class='rp'>{line}</p>"
        return html

    def skill_lines(raw):
        html = ""
        for line in raw.split('\n'):
            line = line.strip()
            if not line:
                continue
            if line.startswith("-"):
                parts = line.lstrip("- ").split(":", 1)
                if len(parts) == 2:
                    html += f"<div class='bul'><span class='bd'>-</span> <b>{parts[0].strip()}:</b> {parts[1].strip()}</div>"
                else:
                    html += f"<div class='bul'><span class='bd'>-</span> {line.lstrip('- ').strip()}</div>"
        return html

    def project_blocks(raw):
        html = ""
        current_name = ""
        current_bullets = []
        for line in raw.split('\n'):
            line = line.strip()
            if not line:
                continue
            if line.upper().startswith("PROJECT:"):
                if current_name:
                    html += f"<p class='pname'>{current_name}</p>"
                    for b in current_bullets:
                        html += f"<div class='bul'><span class='bd'>-</span> {b}</div>"
                    html += "<div style='height:6px;'></div>"
                current_name = line.split(":", 1)[1].strip()
                current_bullets = []
            elif line.startswith("-"):
                current_bullets.append(line.lstrip("- ").strip())
        if current_name:
            html += f"<p class='pname'>{current_name}</p>"
            for b in current_bullets:
                html += f"<div class='bul'><span class='bd'>-</span> {b}</div>"
        return html

    def edu_blocks(raw):
        html = ""
        for line in raw.split('\n'):
            line = line.strip()
            if not line:
                html += "<div style='height:5px;'></div>"
                continue
            if line.upper().startswith("BTECH:"):
                parts = line.split(":", 1)[1].strip().split("|")
                degree = parts[0].strip() if len(parts) > 0 else ""
                college = parts[1].strip() if len(parts) > 1 else ""
                years = parts[2].strip() if len(parts) > 2 else ""
                cgpa = parts[3].strip() if len(parts) > 3 else ""
                html += f"<div class='edu-row'><b class='etitle'>{degree}</b></div>"
                html += f"<div class='edu-row'><span>{college}</span><span>{years}</span></div>"
                if cgpa:
                    html += f"<div class='rp'>{cgpa}</div>"
            elif line.upper().startswith("CLASS12:"):
                parts = line.split(":", 1)[1].strip().split("|")
                school = parts[0].strip() if len(parts) > 0 else ""
                year = parts[1].strip() if len(parts) > 1 else ""
                pct = parts[2].strip() if len(parts) > 2 else ""
                html += f"<div class='edu-row'><b class='etitle'>Intermediate (Class 12)</b></div>"
                html += f"<div class='edu-row'><span>{school}</span><span>{year}</span></div>"
                if pct:
                    html += f"<div class='rp'>{pct}</div>"
            elif line.upper().startswith("CLASS10:"):
                parts = line.split(":", 1)[1].strip().split("|")
                school = parts[0].strip() if len(parts) > 0 else ""
                year = parts[1].strip() if len(parts) > 1 else ""
                pct = parts[2].strip() if len(parts) > 2 else ""
                html += f"<div class='edu-row'><b class='etitle'>Secondary School Certificate (Class 10)</b></div>"
                html += f"<div class='edu-row'><span>{school}</span><span>{year}</span></div>"
                if pct:
                    html += f"<div class='rp'>{pct}</div>"
        return html

    html = f"""
    <style>
    .resume-a4 {{
        background: #fff;
        color: #111;
        font-family: Arial, sans-serif;
        font-size: 12px;
        line-height: 1.65;
        padding: 48px 56px;
        max-width: 794px;
        min-height: 1000px;
        margin: 0 auto;
        border: 1px solid #ccc;
        border-radius: 3px;
        box-shadow: 0 2px 12px rgba(0,0,0,0.08);
    }}
    .rname {{ font-size: 22px; font-weight: bold; text-align: center; margin: 0 0 4px; color: #000; }}
    .rcontact {{ display: flex; justify-content: center; gap: 24px; font-size: 11px; color: #444; flex-wrap: wrap; margin-bottom: 4px; }}
    .rcontact span {{ display: inline-block; }}
    .rtopline {{ border: none; border-top: 1px solid #bbb; margin: 8px 0 14px; }}
    .rsec {{ font-size: 11px; font-weight: bold; letter-spacing: 0.06em; color: #000; margin: 14px 0 2px; text-transform: uppercase; border-bottom: 1.2px solid #000; padding-bottom: 2px; }}
    .rp {{ font-size: 11.5px; color: #222; margin: 3px 0; line-height: 1.7; }}
    .bul {{ display: flex; gap: 6px; margin: 2px 0; font-size: 11.5px; color: #222; line-height: 1.65; }}
    .bd {{ flex-shrink: 0; }}
    .pname {{ font-weight: bold; font-size: 11.5px; color: #000; margin: 6px 0 2px; }}
    .edu-row {{ display: flex; justify-content: space-between; font-size: 11.5px; color: #222; margin: 1px 0; }}
    .etitle {{ font-size: 11.5px; color: #000; }}
    .skcat {{ font-weight: bold; }}
    </style>

    <div class='resume-a4'>
      <p class='rname'>{name}</p>
      <div class='rcontact'>
        {'<span>Email: ' + email + '</span>' if email else ''}
        {'<span>Phone: ' + phone + '</span>' if phone else ''}
        {'<span>LinkedIn: ' + linkedin + '</span>' if linkedin else ''}
        {'<span>GitHub: ' + github + '</span>' if github else ''}
      </div>
      <hr class='rtopline'>

      <div class='rsec'>Professional Summary</div>
      <p class='rp'>{summary}</p>

      <div class='rsec'>Technical Skills</div>
      {skill_lines(skills_raw)}

      <div class='rsec'>Projects</div>
      {project_blocks(projects_raw)}

      <div class='rsec'>Education</div>
      {edu_blocks(education_raw)}

      <div class='rsec'>Achievements and Certifications</div>
      {bullet_lines(achievements_raw)}

      <div class='rsec'>Languages Known</div>
      {bullet_lines(languages_raw)}

      <div class='rsec'>Conclusion</div>
      <p class='rp'>{conclusion}</p>
    </div>
    """
    return html

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
    jd_text = st.text_area("Paste the full job description here", height=300,
                           placeholder="e.g. We are looking for a Python developer with...")
with col2:
    st.subheader("Resume")
    uploaded_file = st.file_uploader("Upload resume as PDF", type=["pdf"])
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
                "Here is a professionally rewritten ATS-optimized version:"
            )

            with st.spinner("Generating your improved resume... please wait 20-30 seconds..."):
                try:
                    improved_resume = generate_improved_resume(
                        resume_text, jd_text, missing_keywords
                    )

                    # Display formatted A4
                    display_html = build_display_html(improved_resume)
                    st.markdown(display_html, unsafe_allow_html=True)

                    st.divider()

                    # PDF Download
                    st.subheader("⬇ Download Generated Resume as PDF")
                    st.markdown("Click below to download your resume as a ready-to-send PDF.")

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
                        "or copy the text into Word or Google Docs, format, and save as PDF."
                    )

                except Exception as e:
                    st.error(f"Could not generate resume. Error: {str(e)}")