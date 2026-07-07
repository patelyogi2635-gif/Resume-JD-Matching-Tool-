# Resume ↔ Job Description Matcher

An AI-powered Resume–JD matching application that analyzes resumes against job descriptions using Large Language Models (LLMs), semantic embeddings, and deterministic scoring.

The system extracts skills, computes an explainable match score, measures semantic similarity, and generates recruiter-style feedback. It is designed with a modular backend, reproducible scoring, and a lightweight frontend for easy deployment.

---

## Features

- Extracts required and preferred skills from job descriptions using LLMs.
- Detects demonstrated skills from resumes.
- Computes a deterministic match score in Python.
- Calculates semantic similarity using embedding vectors and cosine similarity.
- Generates recruiter-style feedback aligned with the computed score.
- Supports both pasted text and PDF/TXT uploads.
- Single-service deployment (FastAPI + static frontend).

---

## Scoring Methodology

The final match score combines three independent signals:

| Component | Weight |
|-----------|--------:|
| Must-have skill match | **55%** |
| Nice-to-have skill match | **25%** |
| Semantic similarity | **20%** |

### Skill Matching

Skills are extracted using an LLM and matched using fuzzy matching to account for naming variations.

Examples:

- PostgreSQL ↔ Postgres
- Fast API ↔ FastAPI
- JS ↔ JavaScript

### Semantic Similarity

The complete resume and job description are converted into embedding vectors.

The system computes cosine similarity between both embeddings to measure contextual alignment beyond keyword overlap.

If embeddings are unavailable, the application automatically falls back to skill-only scoring to ensure uninterrupted execution.

---

## Architecture

```
Resume
      │
      ▼
File Parser
      │
      ▼
LLM Skill Extraction
      │
      ▼
Deterministic Scoring Engine
      │
      ├────────► Semantic Embeddings
      │               │
      │               ▼
      │        Cosine Similarity
      │
      ▼
Recruiter Narrative Generator
      │
      ▼
Final JSON Response
```

---

## Project Structure

```
resume-jd-matcher/
│
├── backend/
│   ├── app/
│   │   ├── config.py
│   │   ├── file_parsing.py
│   │   ├── llm_client.py
│   │   ├── pipeline.py
│   │   ├── prompts.py
│   │   ├── schemas.py
│   │   └── scoring.py
│   │
│   ├── main.py
│   ├── debug_connectivity.py
│   ├── requirements.txt
│   └── .env.example
│
├── frontend/
│   └── index.html
│
└── README.md
```

---

## Tech Stack

### Backend

- Python
- FastAPI
- Pydantic

### AI

- Groq LLM
- Google Gemini
- Embeddings
- Cosine Similarity

### Frontend

- HTML
- CSS
- Vanilla JavaScript

### File Processing

- PyMuPDF
- TXT Parsing

---

## API

### POST `/api/match`

Accepts either pasted text or uploaded files.

### Request

| Field | Type |
|------|------|
| resume_text | string |
| resume_file | PDF / TXT |
| jd_text | string |
| jd_file | PDF / TXT |

---

### Example Response

```json
{
  "match_score": 84,
  "semantic_similarity": 81.6,
  "must_have": {
    "matched": [
      "Python",
      "FastAPI",
      "REST APIs"
    ],
    "missing": [
      "Redis"
    ]
  },
  "nice_to_have": {
    "matched": [
      "Docker",
      "LangChain"
    ],
    "missing": [
      "Kubernetes"
    ]
  },
  "resume_skills_detected": [
    "Python",
    "FastAPI",
    "Docker"
  ],
  "narrative": "The candidate demonstrates strong alignment with the role..."
}
```

---

## Local Setup

### Clone the repository

```bash
git clone <repository-url>
cd resume-jd-matcher/backend
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Configure environment variables

```bash
cp .env.example .env
```

Add your API keys:

```
GROQ_API_KEY=...
GEMINI_API_KEY=...
```

### Verify API connectivity

```bash
python debug_connectivity.py
```

### Run the application

```bash
python main.py
```

Backend

```
http://localhost:8000
```

Frontend

```
http://localhost:8000
```

---

## Deployment

The project is designed as a **single FastAPI service**, where:

- FastAPI serves the REST API
- Static frontend is served by FastAPI
- Suitable for Railway, Render, or similar platforms

No frontend build process is required.

---

## Design Principles

- Modular architecture
- Deterministic scoring (not dependent on LLM calculations)
- Explainable recruiter feedback
- Graceful fallback when embeddings are unavailable
- Production-friendly API design
- Lightweight frontend with no framework dependencies

---

## Future Improvements

- Resume section weighting (Projects > Skills > Summary)
- Skill synonym normalization
- DOCX resume support
- Recruiter dashboard
- Batch resume screening
- Caching of LLM responses
- Per-skill confidence scores
- Authentication and usage analytics

---

## License

This project is provided for educational and portfolio purposes.