"""Pydantic models shared across the app."""

from typing import List, Optional

from pydantic import BaseModel


class SkillBreakdown(BaseModel):
    matched: List[str]
    missing: List[str]


class MatchResponse(BaseModel):
    match_score: int
    must_have: SkillBreakdown
    nice_to_have: SkillBreakdown
    semantic_similarity: Optional[float] = None  # 0-100, None if embedding failed
    resume_skills_detected: List[str]
    narrative: str


class ExtractionResult(BaseModel):
    jd_must_have: List[str] = []
    jd_nice_to_have: List[str] = []
    resume_skills: List[str] = []
