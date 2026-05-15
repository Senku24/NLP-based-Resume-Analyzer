# 🎯 AI Resume Analyzer & Job Match System

An NLP-powered resume analysis prototype built with Python and Streamlit.  
Upload a resume, paste a job description, and get instant skill matching, ATS scoring, and improvement suggestions.

---

## ✨ Features

| Feature | Description |
|---|---|
| 📄 Resume Parsing | Supports PDF, DOCX, and TXT formats |
| 🔬 NLP Pipeline | Tokenization → Stopword Removal → Lemmatization |
| 🛠️ Skill Extraction | Matches 120+ skills across 10 categories |
| 📊 TF-IDF Matching | Cosine similarity for resume–JD comparison |
| 🎯 ATS Score | Weighted score (similarity + skills + completeness) |
| 💡 Suggestions | Actionable improvement recommendations |
| 📈 Visualizations | Gauge charts, bar charts, donut charts (Plotly) |

---

## 🏗️ Project Structure

```
NLP-based-Resume-Analyzer/
├── app.py                  ← Main Streamlit app
├── requirements.txt        ← Python dependencies
├── setup.sh                ← One-command setup script
│
├── data/
│   └── skills.csv          ← Skill dataset (120+ skills)
│
├── utils/
│   ├── parser.py           ← Resume text extraction (PDF/DOCX/TXT)
│   ├── preprocess.py       ← NLP preprocessing pipeline
│   ├── skill_extractor.py  ← Skill matching engine
│   ├── similarity.py       ← TF-IDF cosine similarity
│   ├── ats_score.py        ← ATS score computation
│   └── suggestions.py      ← Recommendation engine
│
└── sample_data/
    ├── resume_software_engineer.txt
    ├── resume_data_scientist.txt
    ├── jd_backend_engineer.txt
    └── jd_data_scientist.txt
```

---

## ⚙️ Setup & Installation

### Step 1 — Clone / Open the project

```bash
cd "NLP-based-Resume-Analyzer"
```

### Step 2 — Run setup (installs all dependencies)

```bash
bash setup.sh
```

This will:
1. Install all Python packages from `requirements.txt`
2. Download the spaCy English model (`en_core_web_sm`)
3. Download required NLTK data (stopwords, punkt, wordnet)

### Step 3 — Launch the app

```bash
streamlit run app.py
```

The app opens at **http://localhost:8501**

---

## 🧪 Manual Testing Guide

Use the provided sample files in `sample_data/` to test:

### Test 1 — High Match (Expected ~75–90%)
- Resume: `resume_software_engineer.txt`
- JD: `jd_backend_engineer.txt`

### Test 2 — High Match (Expected ~80–92%)
- Resume: `resume_data_scientist.txt`
- JD: `jd_data_scientist.txt`

### Test 3 — Lower Match (Expected ~30–55%)
- Resume: `resume_software_engineer.txt`
- JD: `jd_data_scientist.txt`

---

## 🔬 NLP Pipeline

```
Raw Resume Text
    ↓ Lowercase conversion
    ↓ Punctuation & special char removal
    ↓ Tokenization (spaCy)
    ↓ Stopword removal (NLTK + spaCy)
    ↓ Lemmatization (spaCy)
    ↓ Skill phrase matching
    ↓ TF-IDF vectorization (scikit-learn)
    ↓ Cosine similarity computation
    ↓ Weighted ATS score
    ↓ Dashboard visualization
```

---

## 📐 ATS Score Formula

| Component | Weight |
|---|---|
| TF-IDF Cosine Similarity | 50% |
| Skill Overlap Ratio | 30% |
| Resume Completeness Signals | 20% |

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| UI | Streamlit |
| NLP | spaCy, NLTK |
| ML | scikit-learn (TF-IDF) |
| Visualization | Plotly |
| File Parsing | pdfplumber, python-docx |
| Language | Python 3.9+ |

---

## 📋 Requirements

- Python 3.9 or higher
- Internet connection for initial setup (model downloads)
- ~500 MB disk space for NLP models

---

*Built as an academic prototype MVP — NLP-based Resume Analyzer v1.0*