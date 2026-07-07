"""Orchestrates the full match pipeline. Kept separate from the FastAPI
route so it's independently testable and the route itself stays thin.

Pipeline order matters:
1. Extract structured skills (LLM)
2. Compute deterministic skill-overlap score (Python)
3. Compute semantic similarity via embeddings (LLM), degrading gracefully
4. Blend into final score (Python)
5. Generate narrative from the FINAL score + breakdown (LLM) - so the
   wording can't drift from the number.
"""

import logging

from app.config import MAX_EMBEDDING_CHARS, MAX_JD_CHARS, MAX_RESUME_CHARS
from app.llm_client import LLMCallError, chat_json, chat_text, get_embedding
from app.prompts import EXTRACTION_PROMPT, NARRATIVE_PROMPT
from app.schemas import ExtractionResult, MatchResponse
from app.scoring import compute_match_score, cosine_similarity, score_skills,expand_implied_skills

logger = logging.getLogger("resume-jd-matcher.pipeline")


def fit_level_for_score(score: int) -> str:
    if score >= 75:
        return "strong"
    if score >= 50:
        return "moderate"
    if score >= 25:
        return "weak"
    return "very poor"


def extract_skills(resume_text: str, jd_text: str) -> ExtractionResult:
    prompt = EXTRACTION_PROMPT.format(
        resume=resume_text[:MAX_RESUME_CHARS],
        jd=jd_text[:MAX_JD_CHARS],
    )
    try:
        raw = chat_json(prompt, temperature=0.2, max_tokens=1000)
    except LLMCallError as exc:
        raise RuntimeError(f"Skill extraction failed: {exc}") from exc
    return ExtractionResult(**raw)


def compute_semantic_similarity(resume_text: str, jd_text: str):
    """Returns a 0-100 similarity score, or None if embeddings are unavailable."""
    try:
        resume_vec = get_embedding(resume_text[:MAX_EMBEDDING_CHARS])
        jd_vec = get_embedding(jd_text[:MAX_EMBEDDING_CHARS])
        similarity = cosine_similarity(resume_vec, jd_vec)
        return similarity
    except LLMCallError as exc:
        logger.warning("Semantic similarity unavailable, falling back to skill-only scoring: %s", exc)
        return None


def generate_narrative(score: int, must_have, nice_to_have, semantic_similarity) -> str:
    semantic_note = (
        f"{round(semantic_similarity * 100)}% contextual overlap"
        if semantic_similarity is not None
        else "unavailable for this run"
    )
    prompt = NARRATIVE_PROMPT.format(
        score=score,
        fit_level=fit_level_for_score(score),
        must_matched=", ".join(must_have.matched) or "none",
        must_missing=", ".join(must_have.missing) or "none",
        nice_matched=", ".join(nice_to_have.matched) or "none",
        nice_missing=", ".join(nice_to_have.missing) or "none",
        semantic_note=semantic_note,
    )
    try:
        return chat_text(prompt, temperature=0.3, max_tokens=400)
    except LLMCallError as exc:
        return f"(Narrative generation failed: {exc})"


def run_match_pipeline(resume_text: str, jd_text: str) -> MatchResponse:
    extraction = extract_skills(resume_text, jd_text)
    extraction.resume_skills = expand_implied_skills(extraction.resume_skills)

    must_have_breakdown = score_skills(extraction.jd_must_have, extraction.resume_skills)
    nice_to_have_breakdown = score_skills(extraction.jd_nice_to_have, extraction.resume_skills)

    semantic_similarity = compute_semantic_similarity(resume_text, jd_text)

    final_score = compute_match_score(must_have_breakdown, nice_to_have_breakdown, semantic_similarity)

    narrative = generate_narrative(final_score, must_have_breakdown, nice_to_have_breakdown, semantic_similarity)

    return MatchResponse(
        match_score=final_score,
        must_have=must_have_breakdown,
        nice_to_have=nice_to_have_breakdown,
        semantic_similarity=round(semantic_similarity * 100, 1) if semantic_similarity is not None else None,
        resume_skills_detected=extraction.resume_skills,
        narrative=narrative,
    )
