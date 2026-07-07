"""All scoring logic is deterministic Python - never trust an LLM's own
arithmetic for the final number. The LLM only supplies extracted skill
lists and (separately) narrative text; everything numeric happens here so
it's auditable, testable, and reproducible across runs.
"""

import re
from difflib import SequenceMatcher
from typing import List, Optional

from app.config import (
    FALLBACK_MUST_HAVE_WEIGHT,
    FALLBACK_NICE_TO_HAVE_WEIGHT,
    FUZZY_MATCH_THRESHOLD,
    MUST_HAVE_WEIGHT,
    NICE_TO_HAVE_WEIGHT,
    SEMANTIC_WEIGHT,
)
from app.schemas import SkillBreakdown

# Maps a specific tool/technique to the broader category it demonstrates.
# Fixes cases where a resume lists "FAISS" but the JD asks for "Vector
# databases" - textually unrelated strings, but the same actual skill.
SKILL_IMPLICATIONS = {
    "faiss": ["vector databases", "embeddings"],
    "chromadb": ["vector databases", "embeddings"],
    "pinecone": ["vector databases", "embeddings"],
    "langchain": ["llms", "large language models"],
    "langgraph": ["llms", "large language models", "agents"],
    "prompt engineering": ["llms", "large language models"],
    "groq": ["llms", "large language models"],
    "rag": ["embeddings", "vector databases", "llms", "retrieval augmented generation"],
}


def expand_implied_skills(resume_skills: list) -> list:
    """Add category-level skills implied by specific tools the resume lists."""
    expanded = list(resume_skills)
    normalized_existing = {normalize(s) for s in resume_skills}
    for skill in resume_skills:
        implied = SKILL_IMPLICATIONS.get(normalize(skill), [])
        for imp in implied:
            if imp not in normalized_existing:
                expanded.append(imp)
                normalized_existing.add(imp)
    return expanded


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


def cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
    """Plain-Python cosine similarity - no numpy dependency needed for two
    vectors of a few hundred/thousand floats."""
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = sum(a * a for a in vec_a) ** 0.5
    norm_b = sum(b * b for b in vec_b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def compute_match_score(
    must_have: SkillBreakdown,
    nice_to_have: SkillBreakdown,
    semantic_similarity: Optional[float] = None,
) -> int:
    """Weighted blend of deterministic skill overlap + (optionally) embedding
    cosine similarity. If semantic_similarity is None (embedding call failed),
    its weight is redistributed back into must/nice-have so the score still
    sums correctly without the LLM-embedding signal.
    """
    must_total = len(must_have.matched) + len(must_have.missing)
    nice_total = len(nice_to_have.matched) + len(nice_to_have.missing)

    must_have_ratio = (len(must_have.matched) / must_total) if must_total else 1.0
    nice_have_ratio = (len(nice_to_have.matched) / nice_total) if nice_total else 1.0

    if semantic_similarity is None:
        must_weight = FALLBACK_MUST_HAVE_WEIGHT
        nice_weight = FALLBACK_NICE_TO_HAVE_WEIGHT
        semantic_score = 0.0
    else:
        must_weight = MUST_HAVE_WEIGHT
        nice_weight = NICE_TO_HAVE_WEIGHT
        # cosine similarity is typically 0.3-0.9 in practice even for decent
        # matches, so clamp to [0,1] defensively - the LLM/embedding API
        # occasionally returns values fractionally outside range due to
        # floating point.
        semantic_score = max(0.0, min(1.0, semantic_similarity)) * SEMANTIC_WEIGHT

    total = (must_have_ratio * must_weight) + (nice_have_ratio * nice_weight) + semantic_score
    return round(total)
