"""
Resume-JD Matching Tool - Backend
E2M Solutions AI Engineering Take-Home Assessment

Architecture:
- FastAPI handles file upload / text input, parsing, and orchestration
- One LLM call extracts structured skills from the resume and JD (Groq
  primary, Gemini fallback via LiteLLM)
- The MATCH SCORE ITSELF is computed deterministically in Python, not by the
  LLM. LLMs are unreliable at arithmetic and inconsistent across runs, so we
  only trust them for extraction and narrative writing (subjective language
  tasks) and do the scoring with plain, auditable, testable code.
- The narrative is generated in a SECOND LLM call, made AFTER the score is
  computed, and is given the score + skill breakdown directly - so the
  wording is forced to agree with the number instead of being written blind.
"""

import io
import json
import logging
import os
import re
from difflib import SequenceMatcher
from typing import List, Optional

import pdfplumber
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from litellm import completion
from pydantic import BaseModel

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("resume-jd-matcher")

app = FastAPI(title="Resume-JD Matching Tool")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tightened in production; open for local assessment demo
    allow_methods=["*"],
    allow_headers=["*"],
)

GROQ_MODEL = "groq/llama-3.3-70b-versatile"
GEMINI_MODEL = "gemini/gemini-1.5-flash"

MUST_HAVE_WEIGHT = 70
NICE_TO_HAVE_WEIGHT = 30
FUZZY_MATCH_THRESHOLD = 0.82  # e.g. "Postgres" vs "PostgreSQL"


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class SkillBreakdown(BaseModel):
    matched: List[str]
    missing: List[str]


class MatchResponse(BaseModel):
    match_score: int
    must_have: SkillBreakdown
    nice_to_have: SkillBreakdown
    resume_skills_detected: List[str]
    narrative: str


# ---------------------------------------------------------------------------
# Text extraction
# ---------------------------------------------------------------------------

def extract_text_from_upload(file_bytes: bytes, filename: str) -> str:
    """Extract plain text from an uploaded file. Falls back to raw decode
    for non-PDF files (txt, md, etc.)."""
    if filename.lower().endswith(".pdf"):
        text_parts = []
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(page_text)
        return "\n".join(text_parts)
    try:
        return file_bytes.decode("utf-8", errors="ignore")
    except Exception:
        return ""


async def resolve_text(text_field: Optional[str], file_field: Optional[UploadFile]) -> str:
    """A given input (resume or JD) can arrive as pasted text or an uploaded
    file. Prefer the uploaded file if both are present."""
    if file_field is not None and file_field.filename:
        raw = await file_field.read()
        extracted = extract_text_from_upload(raw, file_field.filename)
        if extracted.strip():
            return extracted
    if text_field and text_field.strip():
        return text_field
    return ""


# ---------------------------------------------------------------------------
# LLM prompts
# ---------------------------------------------------------------------------

EXTRACTION_PROMPT = """You are an expert technical recruiter and skills-taxonomy analyst.

Given a RESUME and a JOB DESCRIPTION, extract:

1. "jd_must_have": explicitly required, core, or non-negotiable skills from the JD
2. "jd_nice_to_have": preferred, bonus, or "plus" skills from the JD
   If the JD doesn't clearly separate these, use your judgement based on language
   like "required" vs "preferred"/"nice to have"/"bonus". Default ambiguous items
   to must_have if the JD emphasizes them, otherwise nice_to_have.
3. "resume_skills": every skill, tool, language, framework, or relevant
   qualification actually demonstrated in the resume (via experience,
   projects, or explicit skills sections) - not just keywords.

Respond with ONLY valid JSON, no markdown fences, no preamble, in exactly
this shape:
{{
  "jd_must_have": ["skill1", "skill2"],
  "jd_nice_to_have": ["skill1", "skill2"],
  "resume_skills": ["skill1", "skill2"]
}}

RESUME:
---
{resume}
---

JOB DESCRIPTION:
---
{jd}
---
"""


NARRATIVE_PROMPT = """You are an expert technical recruiter writing honest,
direct feedback for a candidate.

Here is the computed match assessment for a candidate against a job description:

- Overall match score: {score}/100 (this score is ALREADY COMPUTED and correct - do not contradict it or imply a different score)
- Fit level: {fit_level}
- Must-have skills matched: {must_matched}
- Must-have skills MISSING: {must_missing}
- Nice-to-have skills matched: {nice_matched}
- Nice-to-have skills missing: {nice_missing}

Write a 3-5 sentence narrative assessment that is fully consistent with a
{fit_level} ({score}/100) fit. If the fit is weak, say so plainly and do not
soften it with generic praise - lead with the gap, not the strengths. If the
fit is strong, lead with the strengths. End with one concrete, specific
suggestion for closing the biggest missing must-have skill(s).

Respond with plain text only, no JSON, no markdown.
"""


def clean_json_response(raw: str) -> dict:
    """Strip markdown code fences etc. and parse JSON. Raises on failure."""
    cleaned = raw.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    return json.loads(cleaned)


def call_llm_for_extraction(resume_text: str, jd_text: str) -> dict:
    prompt = EXTRACTION_PROMPT.format(resume=resume_text[:12000], jd=jd_text[:6000])
    messages = [{"role": "user", "content": prompt}]

    last_error = None
    for model in (GROQ_MODEL, GEMINI_MODEL):
        try:
            response = completion(
                model=model,
                messages=messages,
                temperature=0.2,
                max_tokens=1000,
            )
            content = response["choices"][0]["message"]["content"]
            return clean_json_response(content)
        except Exception as exc:  # noqa: BLE001
            logger.warning("LLM call failed for model %s: %s", model, exc)
            last_error = exc
            continue

    raise HTTPException(
        status_code=502,
        detail=f"Both Groq and Gemini extraction calls failed: {last_error}",
    )


def fit_level_for_score(score: int) -> str:
    if score >= 75:
        return "strong"
    if score >= 50:
        return "moderate"
    if score >= 25:
        return "weak"
    return "very poor"


def call_llm_for_narrative(score: int, must_have: "SkillBreakdown", nice_to_have: "SkillBreakdown") -> str:
    prompt = NARRATIVE_PROMPT.format(
        score=score,
        fit_level=fit_level_for_score(score),
        must_matched=", ".join(must_have.matched) or "none",
        must_missing=", ".join(must_have.missing) or "none",
        nice_matched=", ".join(nice_to_have.matched) or "none",
        nice_missing=", ".join(nice_to_have.missing) or "none",
    )
    messages = [{"role": "user", "content": prompt}]

    last_error = None
    for model in (GROQ_MODEL, GEMINI_MODEL):
        try:
            response = completion(
                model=model,
                messages=messages,
                temperature=0.3,
                max_tokens=400,
            )
            return response["choices"][0]["message"]["content"].strip()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Narrative LLM call failed for model %s: %s", model, exc)
            last_error = exc
            continue

    return f"(Narrative generation failed: {last_error})"


# ---------------------------------------------------------------------------
# Deterministic scoring
# ---------------------------------------------------------------------------

def normalize(skill: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", skill.lower()).strip()


def is_match(required_skill: str, resume_skills_normalized: List[str]) -> bool:
    req_norm = normalize(required_skill)
    for rs in resume_skills_normalized:
        if req_norm == rs or req_norm in rs or rs in req_norm:
            return True
        if SequenceMatcher(None, req_norm, rs).ratio() >= FUZZY_MATCH_THRESHOLD:
            return True
    return False


def score_skills(required: List[str], resume_skills: List[str]) -> SkillBreakdown:
    resume_norm = [normalize(s) for s in resume_skills]
    matched, missing = [], []
    for skill in required:
        (matched if is_match(skill, resume_norm) else missing).append(skill)
    return SkillBreakdown(matched=matched, missing=missing)


def compute_match_score(must_have: SkillBreakdown, nice_to_have: SkillBreakdown) -> int:
    must_total = len(must_have.matched) + len(must_have.missing)
    nice_total = len(nice_to_have.matched) + len(nice_to_have.missing)

    must_score = (len(must_have.matched) / must_total * MUST_HAVE_WEIGHT) if must_total else MUST_HAVE_WEIGHT
    nice_score = (len(nice_to_have.matched) / nice_total * NICE_TO_HAVE_WEIGHT) if nice_total else NICE_TO_HAVE_WEIGHT

    return round(must_score + nice_score)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/api/health")
async def health():
    return {"status": "ok"}


@app.post("/api/match", response_model=MatchResponse)
async def match_resume_to_jd(
    resume_text: Optional[str] = Form(None),
    jd_text: Optional[str] = Form(None),
    resume_file: Optional[UploadFile] = File(None),
    jd_file: Optional[UploadFile] = File(None),
):
    resume = await resolve_text(resume_text, resume_file)
    jd = await resolve_text(jd_text, jd_file)

    if not resume.strip():
        raise HTTPException(status_code=400, detail="No resume text or file provided.")
    if not jd.strip():
        raise HTTPException(status_code=400, detail="No job description text or file provided.")

    extraction = call_llm_for_extraction(resume, jd)

    jd_must_have = extraction.get("jd_must_have", [])
    jd_nice_to_have = extraction.get("jd_nice_to_have", [])
    resume_skills = extraction.get("resume_skills", [])

    must_have_breakdown = score_skills(jd_must_have, resume_skills)
    nice_to_have_breakdown = score_skills(jd_nice_to_have, resume_skills)
    final_score = compute_match_score(must_have_breakdown, nice_to_have_breakdown)

    narrative = call_llm_for_narrative(final_score, must_have_breakdown, nice_to_have_breakdown)

    return MatchResponse(
        match_score=final_score,
        must_have=must_have_breakdown,
        nice_to_have=nice_to_have_breakdown,
        resume_skills_detected=resume_skills,
        narrative=narrative,
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)