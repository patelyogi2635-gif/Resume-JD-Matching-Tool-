"""
Run this FIRST, before starting the server, to confirm both API keys
actually work: `python debug_connectivity.py`

Tests three things independently so you know exactly which piece is broken
if something fails:
1. Groq chat completion
2. Gemini chat completion (fallback path)
3. Gemini embeddings (needed for the semantic similarity feature)
"""

import os
import sys

from dotenv import load_dotenv

load_dotenv()

from litellm import completion, embedding  # noqa: E402

GROQ_MODEL = "groq/llama-3.3-70b-versatile"
GEMINI_MODEL = "gemini/gemini-2.5-flash"
GEMINI_EMBEDDING_MODEL = "gemini/gemini-embedding-001"


def check_env_vars():
    print("--- Checking environment variables ---")
    groq_key = os.environ.get("GROQ_API_KEY")
    gemini_key = os.environ.get("GEMINI_API_KEY")

    if not groq_key:
        print("[FAIL] GROQ_API_KEY is not set (check your .env file)")
    else:
        print(f"[OK] GROQ_API_KEY is set (starts with {groq_key[:6]}...)")

    if not gemini_key:
        print("[FAIL] GEMINI_API_KEY is not set (check your .env file)")
    else:
        print(f"[OK] GEMINI_API_KEY is set (starts with {gemini_key[:6]}...)")

    print()
    return bool(groq_key), bool(gemini_key)


def test_groq_chat():
    print("--- Testing Groq chat completion ---")
    try:
        response = completion(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": "Reply with exactly: OK"}],
            max_tokens=10,
        )
        content = response["choices"][0]["message"]["content"]
        print(f"[OK] Groq responded: {content!r}")
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"[FAIL] Groq chat completion failed:\n  {exc}")
        return False


def test_gemini_chat():
    print("\n--- Testing Gemini chat completion ---")
    try:
        response = completion(
            model=GEMINI_MODEL,
            messages=[{"role": "user", "content": "Reply with exactly: OK"}],
            max_tokens=10,
        )
        content = response["choices"][0]["message"]["content"]
        print(f"[OK] Gemini responded: {content!r}")
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"[FAIL] Gemini chat completion failed:\n  {exc}")
        return False


def test_gemini_embedding():
    print("\n--- Testing Gemini embeddings ---")
    try:
        response = embedding(model=GEMINI_EMBEDDING_MODEL, input=["hello world"])
        vec = response["data"][0]["embedding"]
        print(f"[OK] Gemini embedding returned a {len(vec)}-dimensional vector")
        print(f"     First 5 values: {vec[:5]}")
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"[FAIL] Gemini embedding failed:\n  {exc}")
        return False


if __name__ == "__main__":
    has_groq, has_gemini = check_env_vars()

    results = {}
    if has_groq:
        results["groq_chat"] = test_groq_chat()
    else:
        print("--- Skipping Groq chat test (no key) ---\n")
        results["groq_chat"] = False

    if has_gemini:
        results["gemini_chat"] = test_gemini_chat()
        results["gemini_embedding"] = test_gemini_embedding()
    else:
        print("--- Skipping Gemini tests (no key) ---")
        results["gemini_chat"] = False
        results["gemini_embedding"] = False

    print("\n=== SUMMARY ===")
    for name, ok in results.items():
        print(f"{name}: {'PASS' if ok else 'FAIL'}")

    if not results["groq_chat"] and not results["gemini_chat"]:
        print("\nNeither chat model works - the app cannot function until at least one is fixed.")
        sys.exit(1)
    if not results["gemini_embedding"]:
        print("\nEmbeddings unavailable - the app will still work, but will silently"
              " fall back to skill-matching-only scoring (no semantic similarity).")