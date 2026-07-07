"""Thin wrapper around litellm for both chat completion and embeddings.

Chat completion: tries Groq first, falls back to Gemini on any error
(rate limit, timeout, auth, etc).

Embeddings: Groq has no embeddings endpoint, so this always goes through
Gemini (gemini/text-embedding-004). If it fails, callers should treat the
semantic similarity signal as unavailable rather than failing the whole
request - a resume/JD match is still useful without it.
"""

import json
import logging
import re

from litellm import completion, embedding

from app.config import GEMINI_MODEL, GEMINI_EMBEDDING_MODEL, GROQ_MODEL

logger = logging.getLogger("resume-jd-matcher.llm_client")


class LLMCallError(Exception):
    pass


def _strip_json_fences(raw: str) -> str:
    cleaned = raw.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    return cleaned


def chat_json(prompt: str, temperature: float = 0.2, max_tokens: int = 1000) -> dict:
    """Call the chat model expecting a JSON object back. Tries Groq then Gemini."""
    messages = [{"role": "user", "content": prompt}]
    last_error = None

    for model in (GROQ_MODEL, GEMINI_MODEL):
        try:
            response = completion(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,

            )
            content = response["choices"][0]["message"]["content"]
            return json.loads(_strip_json_fences(content))
        except Exception as exc:  # noqa: BLE001
            logger.warning("chat_json failed for model %s: %s", model, exc)
            last_error = exc
            continue

    raise LLMCallError(f"Both Groq and Gemini failed: {last_error}")


def chat_text(prompt: str, temperature: float = 0.3, max_tokens: int = 400) -> str:
    """Call the chat model expecting plain text back. Tries Groq then Gemini."""
    messages = [{"role": "user", "content": prompt}]
    last_error = None

    for model in (GROQ_MODEL, GEMINI_MODEL):
        try:
            response = completion(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,

            )
            return response["choices"][0]["message"]["content"].strip()
        except Exception as exc:  # noqa: BLE001
            logger.warning("chat_text failed for model %s: %s", model, exc)
            last_error = exc
            continue

    raise LLMCallError(f"Both Groq and Gemini failed: {last_error}")


def get_embedding(text: str) -> list:
    """Get a single embedding vector for text via Gemini. Raises LLMCallError
    on failure - caller decides how to degrade gracefully."""
    try:
        response = embedding(model=GEMINI_EMBEDDING_MODEL, input=[text])
        return response["data"][0]["embedding"]
    except Exception as exc:  # noqa: BLE001
        logger.warning("get_embedding failed: %s", exc)
        raise LLMCallError(f"Gemini embedding call failed: {exc}") from exc
