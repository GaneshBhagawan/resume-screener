# AI Resume Screener

An AI-powered web app that compares a resume against a job description,
shows missing keywords, and auto-generates an improved resume if the
match score is below 90%.

## Features
- Upload any resume as a PDF
- Paste any job description
- Get a match score out of 100%
- See missing keywords instantly
- Auto-generate an improved resume tailored to the JD (if score < 90%)

## Tech Stack
- Python 3.9+
- Streamlit (frontend + deployment)
- TF-IDF Vectorizer (scikit-learn)
- pdfplumber (PDF text extraction)
- Claude AI by Anthropic (resume generation)

## Live Demo
[Click here to try it](https://resume-screener-45wnika6sp9p67dysmy7xs.streamlit.app)

## How to run locally
pip install -r requirements.txt
streamlit run app.py