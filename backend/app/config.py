"""Central configuration - all tunable constants live here."""

import os

from dotenv import load_dotenv

load_dotenv()

# --- LLM models -------------------------------------------------------------
GROQ_MODEL = "groq/llama-3.3-70b-versatile"
GEMINI_MODEL = "gemini/gemini-2.5-flash"
GEMINI_EMBEDDING_MODEL = "gemini/gemini-embedding-001"

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

# --- Scoring weights (must sum to 100) --------------------------------------
# Must-have and nice-to-have come from deterministic skill matching.
# Semantic similarity comes from embedding cosine similarity between the
# full resume and full JD text - it captures overall contextual fit that
# keyword/fuzzy matching can miss (e.g. similar responsibilities phrased
# differently).
MUST_HAVE_WEIGHT = 55
NICE_TO_HAVE_WEIGHT = 25
SEMANTIC_WEIGHT = 20

assert MUST_HAVE_WEIGHT + NICE_TO_HAVE_WEIGHT + SEMANTIC_WEIGHT == 100

# If the embedding call fails for any reason, we don't want to fail the whole
# request - we redistribute SEMANTIC_WEIGHT back into must/nice proportionally.
FALLBACK_MUST_HAVE_WEIGHT = 70
FALLBACK_NICE_TO_HAVE_WEIGHT = 30

# --- Fuzzy skill matching ----------------------------------------------------
FUZZY_MATCH_THRESHOLD = 0.82  # e.g. "Postgres" vs "PostgreSQL"

# --- Text truncation (keeps prompts/embeddings within free-tier limits) -----
MAX_RESUME_CHARS = 12000
MAX_JD_CHARS = 6000
MAX_EMBEDDING_CHARS = 8000  # gemini embedding max is ~8192 tokens
