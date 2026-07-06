# Resume ↔ JD Matching Tool

E2M Solutions — AI Engineering Take-Home Assessment (Test 01)

## What it does

Given a resume and a job description (pasted text or uploaded file), the tool:
1. Extracts must-have vs nice-to-have skills from the JD, and demonstrated skills from the resume, using an LLM (Groq primary, Gemini fallback).
2. **Computes the match score deterministically in Python** — not via the LLM — using fuzzy string matching (`difflib`) against the extracted skill lists, weighted 70% must-have / 30% nice-to-have. LLMs are inconsistent at arithmetic and re-running the same input can produce different scores; keeping scoring in code makes it auditable, testable, and reproducible.
3. Returns matched/missing skills per category plus a narrative explanation with improvement suggestions.

## Architecture

```
frontend/index.html   → single-page vanilla JS/HTML, no build step
backend/main.py        → FastAPI, one route: POST /api/match
```

- **Parsing:** `pdfplumber` for PDF uploads; raw text decode for `.txt`.
- **LLM routing:** `litellm` — tries Groq (`llama-3.3-70b-versatile`) first, falls back to Gemini (`gemini-1.5-flash`) if Groq fails/rate-limits.
- **Scoring:** normalized string comparison + `SequenceMatcher` fuzzy ratio (catches things like "Postgres" vs "PostgreSQL").

## Setup

```bash
cd backend
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
# edit .env and add your GROQ_API_KEY and GEMINI_API_KEY
python main.py
```

Backend runs at `http://localhost:8000`. Health check: `GET /api/health`.

Then just open `frontend/index.html` directly in a browser (double-click it, or `open frontend/index.html`). No build step, no npm install.

## API

`POST /api/match` (multipart/form-data)

| Field | Type | Notes |
|---|---|---|
| `resume_text` | string (optional) | Pasted resume text |
| `resume_file` | file (optional) | PDF or .txt upload |
| `jd_text` | string (optional) | Pasted JD text |
| `jd_file` | file (optional) | PDF or .txt upload |

At least one of `resume_text`/`resume_file` and one of `jd_text`/`jd_file` is required. If both text and file are given for the same side, the file wins.

Response:
```json
{
  "match_score": 78,
  "must_have": { "matched": ["Python", "FastAPI"], "missing": ["Kubernetes"] },
  "nice_to_have": { "matched": ["Docker"], "missing": ["GraphQL"] },
  "resume_skills_detected": ["Python", "FastAPI", "Docker", "..."],
  "narrative": "The candidate is a strong match for..."
}
```

## Design notes / trade-offs (given the 24h window)

- No database — the tool is stateless per request, which matches the spec (no persistence requirement).
- Single LLM call per match (extraction + narrative together) rather than multiple chained calls, to stay fast and within free-tier rate limits.
- Frontend is plain HTML/JS instead of a framework build — same functional result, zero build/deploy risk under time pressure.
- Fuzzy matching threshold (0.82) and must/nice weighting (70/30) are configurable constants at the top of `main.py`.

## Possible next steps (if given more time)

- Cache LLM extraction results by resume+JD hash to avoid recomputation.
- Add a confidence score per matched skill (exact vs fuzzy match).
- Support .docx resumes.
- Add streaming response for the narrative so the UI feels faster.
