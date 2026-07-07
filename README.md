# 🚀 Resume ↔ Job Description Matcher

An AI-powered Resume–Job Description matching application that evaluates how well a candidate's resume aligns with a target job description using **LLMs, semantic embeddings, and deterministic scoring**.

The application extracts required skills, computes an explainable match score, measures semantic similarity, and generates recruiter-style feedback to help candidates identify strengths and skill gaps.

## 🌐 Live Demo

**Application:** https://resume-jd-matching-tool-production.up.railway.app/

**API Documentation:** https://resume-jd-matching-tool-production.up.railway.app/docs

**Health Check:** https://resume-jd-matching-tool-production.up.railway.app/api/health

---

# ✨ Features

* 📄 Upload Resume (PDF/TXT) or paste resume text
* 📋 Upload Job Description (PDF/TXT) or paste JD text
* 🤖 AI-powered skill extraction using LLMs
* 📊 Deterministic ATS-style match score
* 🧠 Semantic similarity using embedding vectors
* ✅ Must-have and Nice-to-have skill analysis
* 📌 Recruiter-style narrative explanation
* 🔄 Automatic fallback between AI providers
* 🚀 Single-service deployment with FastAPI and Railway

---

# 🏗️ System Architecture

```
                Resume              Job Description
                   │                      │
                   └──────────┬───────────┘
                              │
                      File/Text Parser
                              │
                              ▼
                   LLM Skill Extraction
                              │
          ┌───────────────────┴───────────────────┐
          │                                       │
          ▼                                       ▼
   Skill Matching Engine                 Semantic Embeddings
          │                                       │
          └──────────────┬────────────────────────┘
                         ▼
              Deterministic Scoring Engine
                         │
                         ▼
            Recruiter Narrative Generator
                         │
                         ▼
                  JSON Response + UI
```

---

# 📊 Scoring Methodology

The final score is computed deterministically in Python instead of relying on the language model.

| Component           |  Weight |
| ------------------- | ------: |
| Must-have Skills    | **55%** |
| Nice-to-have Skills | **25%** |
| Semantic Similarity | **20%** |

## Skill Matching

Uses fuzzy matching to recognize similar technologies.

Examples:

* PostgreSQL ↔ Postgres
* Fast API ↔ FastAPI
* JS ↔ JavaScript

## Semantic Matching

* Gemini Embeddings
* Cosine Similarity

If embedding generation fails, the application automatically falls back to a skill-only scoring strategy, ensuring uninterrupted execution.

---

# 🛠️ Tech Stack

## Backend

* Python
* FastAPI
* Pydantic
* Uvicorn

## AI

* Groq
* Google Gemini
* LiteLLM
* Gemini Embeddings

## Frontend

* HTML
* CSS
* Vanilla JavaScript

## File Processing

* PDF Parsing
* Text Extraction

## Deployment

* Railway

---

# 📂 Project Structure

```
Resume-JD-Matching-Tool
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
│   ├── frontend/
│   ├── main.py
│   ├── requirements.txt
│   └── .env.example
│
├── README.md
└── .gitignore
```

---

# 📡 API

## POST `/api/match`

Accepts either uploaded files or pasted text.

### Request

| Field       | Type      |
| ----------- | --------- |
| resume_text | string    |
| resume_file | PDF / TXT |
| jd_text     | string    |
| jd_file     | PDF / TXT |

### Example Response

```json
{
  "match_score": 89,
  "semantic_similarity": 82.6,
  "must_have": {
    "matched": [
      "Python",
      "FastAPI",
      "LLMs",
      "Docker"
    ],
    "missing": [
      "Redis"
    ]
  },
  "nice_to_have": {
    "matched": [
      "LangChain",
      "LangGraph"
    ],
    "missing": [
      "AWS",
      "CI/CD"
    ]
  },
  "resume_skills_detected": [
    "Python",
    "FastAPI",
    "Docker",
    "LangChain"
  ],
  "narrative": "The candidate demonstrates a strong alignment with the role..."
}
```

---

# ⚙️ Local Setup

## Clone the repository

```bash
git clone https://github.com/patelyogi2635-gif/Resume-JD-Matching-Tool-.git
cd Resume-JD-Matching-Tool/backend
```

## Install dependencies

```bash
pip install -r requirements.txt
```

## Configure environment variables

Create a `.env` file.

```
GROQ_API_KEY=your_groq_api_key
GEMINI_API_KEY=your_gemini_api_key
```

## Start the application

```bash
python main.py
```

The application will be available at:

```
http://localhost:8000
```

---

# 🎯 Design Principles

* Modular architecture
* Deterministic scoring
* Explainable AI outputs
* Semantic matching beyond keyword search
* Fault-tolerant fallback strategy
* Lightweight frontend
* Production-ready deployment

---

# 🔮 Future Improvements

* DOCX resume support
* Skill synonym knowledge base
* Resume section weighting
* Batch resume screening
* Authentication
* User dashboard
* Recruiter analytics
* Cached LLM responses
* Per-skill confidence scores

---

# 📄 License

This project is released for educational and portfolio purposes.
