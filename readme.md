# Resume Shortlisting System using Deep Learning (SBERT + NLP)

[![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.20-orange?logo=tensorflow)](https://www.tensorflow.org/)
[![SBERT](https://img.shields.io/badge/SBERT-all--MiniLM--L6--v2-yellow)](https://www.sbert.net/)
[![spaCy](https://img.shields.io/badge/spaCy-en__core__web__trf-09A3D5?logo=spacy)](https://spacy.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An intelligent resume shortlisting system that combines **Sentence-BERT (SBERT)** semantic embeddings, **spaCy NER-based skill extraction**, and **regex-based experience parsing** to compute a weighted fit score between a candidate's resume and a job description — automatically ranking and shortlisting candidates.

---

## 📌 Table of Contents

- [Overview](#overview)
- [How It Works](#how-it-works)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Scoring Logic](#scoring-logic)
- [Sample Output](#sample-output)
- [Dependencies](#dependencies)

---

## 🧠 Overview

This system automates the resume screening process by:

1. **Extracting text** from PDF resumes using `PyPDF2`
2. **Preprocessing** text (lowercasing, whitespace normalization)
3. **Generating semantic embeddings** with SBERT (`all-MiniLM-L6-v2`)
4. **Extracting named entities** (skills, organizations, products) using spaCy's transformer pipeline (`en_core_web_trf`)
5. **Parsing years of experience** from text using regex
6. **Computing a weighted final fit score** from all three signals

---

## ⚙️ How It Works

```
Resume (PDF / Text)          Job Description (Text)
        │                              │
        ▼                              ▼
  extract_resume_text()          clean_text()
        │                              │
        └──────────┬───────────────────┘
                   ▼
        ┌──────────────────────┐
        │  SBERT Embeddings    │  → cosine_similarity() → Semantic Score (60%)
        └──────────────────────┘
        ┌──────────────────────┐
        │  spaCy NER           │  → skill_match_ratio()  → Skill Score    (25%)
        └──────────────────────┘
        ┌──────────────────────┐
        │  Regex Experience    │  → extract_experience_years() → Exp Score (15%)
        └──────────────────────┘
                   │
                   ▼
          Final Weighted Score
        (Shortlisted if ≥ 70%)
```

---

## 📁 Project Structure

```
resume-shortlisting-sbert/
│
├── Resume_Shortlisting.ipynb    # Main Jupyter Notebook
└── README.md
```

---

## ⚙️ Installation

### Prerequisites

- Python 3.12+
- Jupyter Notebook or JupyterLab

### Install Dependencies

```bash
pip install sentence-transformers spacy PyPDF2 scikit-learn tensorflow tf-keras
```

### Download spaCy Model

```bash
python -m spacy download en_core_web_trf
```

> **Note:** `en_core_web_trf` is a transformer-based spaCy pipeline (~457 MB). Ensure you have a stable internet connection for the first download.

> **Compatibility Warning:** Installing `sentence-transformers` may upgrade NumPy to 2.x, which is incompatible with `mediapipe` and older `tensorflow` builds. If you face conflicts, pin NumPy:
> ```bash
> pip install "numpy<2"
> ```

---

## 🚀 Usage

### Run in Jupyter Notebook

Open `Resume_Shortlisting.ipynb` and run all cells in order.

### Step-by-step in Python

**1. Extract text from a PDF resume:**
```python
from PyPDF2 import PdfReader

def extract_resume_text(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        if page.extract_text():
            text += page.extract_text()
    return text
```

**2. Load models:**
```python
from sentence_transformers import SentenceTransformer
import spacy

sbert_model = SentenceTransformer("all-MiniLM-L6-v2")
nlp = spacy.load("en_core_web_trf")
```

**3. Compute fit score:**
```python
score = final_fit_score(resume_text, job_description)
print("Final Fit Score:", score)

if score >= 70:
    print("Status: SHORTLISTED ✅")
else:
    print("Status: REJECTED ❌")
```

**4. Inspect skill matches:**
```python
resume_skills = extract_skills(resume_text)
jd_skills = extract_skills(job_description)

print("Matched Skills:", set(resume_skills).intersection(jd_skills))
print("Missing Skills:", set(jd_skills) - set(resume_skills))
```

---

## 📊 Scoring Logic

The final fit score is a **weighted combination** of three components:

| Component | Method | Weight |
|-----------|--------|--------|
| Semantic Similarity | Cosine similarity of SBERT embeddings | 60% |
| Skill Match Ratio | NER-extracted skill overlap (ORG, PRODUCT entities via spaCy) | 25% |
| Experience Match | Regex-parsed years of experience, capped at JD requirement | 15% |

```python
final_score = (
    0.6 * semantic_score +
    0.25 * skill_score +
    0.15 * exp_score
)
```

A candidate is **shortlisted** if their final score is **≥ 70**.

---

## 📈 Sample Output

Using a Machine Learning Engineer job description and a matching resume:

```
Final Fit Score: 89.61

Matched Skills: {'pytorch', 'tensorflow'}
Missing Skills: set()
Status: SHORTLISTED ✅
```

---

## 📦 Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `sentence-transformers` | 5.2.3 | SBERT semantic embeddings |
| `spacy` | 3.8.7 | NLP pipeline and NER |
| `en_core_web_trf` | 3.8.0 | Transformer-based spaCy model |
| `PyPDF2` | 3.0.1 | PDF text extraction |
| `scikit-learn` | 1.5.2 | Cosine similarity |
| `tensorflow` | 2.20.0 | Deep learning backend |
| `tf-keras` | 2.20.1 | Keras compatibility layer |
| `numpy` | 2.2.6 | Numerical operations |

---

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.