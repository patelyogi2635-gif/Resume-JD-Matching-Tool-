"""
Resume-JD Matching Tool - Backend entrypoint


Routes only. All logic lives in app/: config, schemas, prompts, llm_client,
scoring, file_parsing, pipeline.
"""

import logging

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from typing import Optional

from app.file_parsing import resolve_text
from app.pipeline import run_match_pipeline
from app.schemas import MatchResponse

logging.basicConfig(level=logging.INFO)

app = FastAPI(title="Resume-JD Matching Tool")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tightened in production; open for local/assessment demo
    allow_methods=["*"],
    allow_headers=["*"],
)


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

    try:
        return run_match_pipeline(resume, jd)
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


# Serve the frontend as static files from the same service, so one Railway
# deployment gives you a single URL for both the API and the UI.
# Must be mounted last so it doesn't shadow the /api/* routes above.
app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")


if __name__ == "__main__":
    import os

    import uvicorn

    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)