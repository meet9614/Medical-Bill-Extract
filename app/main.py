"""
Medical Bill Extractor - FastAPI Application
Bajaj Finserv Health Datathon
"""

import os
import io
import base64
import tempfile
import requests
import mimetypes
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.local_backend import build_extractor
from app.schemas import ExtractionResponse

app = FastAPI(
    title="Medical Bill Extractor API",
    description="Extracts structured line items from hospital bills and invoices using a hybrid OCR + LLM pipeline.",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Backend is chosen by EXTRACTOR_BACKEND (gemini | local); see app/local_backend.py
extractor = build_extractor()


class DocumentURLRequest(BaseModel):
    document: str  # URL to the document


@app.get("/health")
async def health():
    backend = os.getenv("EXTRACTOR_BACKEND", "gemini").lower()
    return {
        "status": "ok",
        "version": "2.0.0",
        "backend": backend,
        "model": (
            os.getenv("VLM_MODEL_ID", "Qwen/Qwen2-VL-2B-Instruct")
            if backend == "local"
            else os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
        ),
    }



@app.post("/extract-from-file", response_model=ExtractionResponse)
async def extract_from_file(file: UploadFile = File(...)):
    """
    Accept a multipart file upload (PDF or image) and extract bill line items.
    """
    content = await file.read()
    suffix = Path(file.filename).suffix if file.filename else ".pdf"
    if not suffix:
        suffix = ".pdf"

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    try:
        result = extractor.extract(tmp_path)
    finally:
        os.unlink(tmp_path)

    return result
