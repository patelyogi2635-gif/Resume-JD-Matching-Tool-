"""Extracting plain text from uploaded resume/JD files."""

import io
from typing import Optional

import pdfplumber
from fastapi import UploadFile


def extract_text_from_upload(file_bytes: bytes, filename: str) -> str:
    """Extract plain text from an uploaded file. Falls back to raw decode
    for non-PDF files (txt, md, etc.) - matches the assessment spec's
    'simple text extraction if no PDF parser' allowance."""
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
