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

from app.extractor import BillExtractor
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

extractor = BillExtractor()


class DocumentURLRequest(BaseModel):
    document: str  # URL to the document


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "version": "2.0.0",
        "model": os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
    }


@app.post("/extract-bill-data", response_model=ExtractionResponse)
async def extract_from_url(request: DocumentURLRequest):
    """
    Accept a document URL (PDF or image), download it, and extract bill line items.
    """
    try:
        response = requests.get(request.document, timeout=30)
        response.raise_for_status()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to download document: {e}")

    content = response.content
    content_type = response.headers.get("Content-Type", "")

    # Detect PDF by magic bytes if content-type is wrong
    if content[:4] == b"%PDF":
        content_type = "application/pdf"

    suffix = ".pdf" if "pdf" in content_type else ".jpg"

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    try:
        result = extractor.extract(tmp_path)
    finally:
        os.unlink(tmp_path)

    return result


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
