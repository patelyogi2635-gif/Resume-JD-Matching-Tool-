"""All LLM prompt templates live here, kept separate from calling code so
they're easy to iterate on without touching orchestration logic."""

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
- Overall contextual/semantic similarity between resume and JD: {semantic_note}

Write a 3-5 sentence narrative assessment that is fully consistent with a
{fit_level} ({score}/100) fit. If the fit is weak, say so plainly and do not
soften it with generic praise - lead with the gap, not the strengths. If the
fit is strong, lead with the strengths. End with one concrete, specific
suggestion for closing the biggest missing must-have skill(s).

Respond with plain text only, no JSON, no markdown.
"""
