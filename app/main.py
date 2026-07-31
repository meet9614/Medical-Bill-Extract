"""
Medical Bill Extractor - FastAPI Application
Bajaj Finserv Health Datathon
"""

import ipaddress
import os
import socket
import tempfile
from pathlib import Path
from urllib.parse import urlparse

import requests
from fastapi import FastAPI, File, HTTPException, UploadFile
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


# Downloads are bounded so a hostile or accidental URL cannot exhaust memory
# or hang a worker.
MAX_DOWNLOAD_BYTES = int(os.getenv("MAX_DOWNLOAD_BYTES", 50 * 1024 * 1024))
DOWNLOAD_TIMEOUT_S = int(os.getenv("DOWNLOAD_TIMEOUT_S", "30"))
ALLOWED_SUFFIXES = {".pdf", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".webp"}


def _assert_public_url(url: str) -> None:
    """
    Reject URLs that point back into the host's own network.

    This endpoint fetches a caller-supplied URL, which is a server-side request
    forgery primitive: without this check, `http://169.254.169.254/...` would let
    a caller read cloud instance credentials through the API, and
    `http://127.0.0.1:8000/` would let them reach services never exposed
    publicly. Resolving the hostname first is essential -- an attacker-controlled
    domain can simply resolve to a private address.
    """
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise HTTPException(400, f"Unsupported URL scheme: {parsed.scheme!r}")
    if not parsed.hostname:
        raise HTTPException(400, "URL has no host")

    try:
        infos = socket.getaddrinfo(parsed.hostname, None)
    except socket.gaierror as e:
        raise HTTPException(400, f"Could not resolve host: {e}") from e

    for info in infos:
        ip = ipaddress.ip_address(info[4][0])
        if (
            ip.is_private or ip.is_loopback or ip.is_link_local
            or ip.is_reserved or ip.is_multicast or ip.is_unspecified
        ):
            raise HTTPException(400, "URL resolves to a non-public address")


def _download(url: str) -> tuple[bytes, str]:
    """Fetch a document, enforcing size and time limits. Returns (bytes, suffix)."""
    _assert_public_url(url)
    try:
        resp = requests.get(url, timeout=DOWNLOAD_TIMEOUT_S, stream=True)
        resp.raise_for_status()
    except requests.RequestException as e:
        raise HTTPException(400, f"Could not fetch document: {e}") from e

    declared = resp.headers.get("Content-Length")
    if declared and int(declared) > MAX_DOWNLOAD_BYTES:
        raise HTTPException(413, "Document exceeds the maximum allowed size")

    # Content-Length is advisory; enforce the cap while streaming too.
    chunks, total = [], 0
    for chunk in resp.iter_content(chunk_size=65536):
        total += len(chunk)
        if total > MAX_DOWNLOAD_BYTES:
            raise HTTPException(413, "Document exceeds the maximum allowed size")
        chunks.append(chunk)
    content = b"".join(chunks)

    suffix = Path(urlparse(url).path).suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        # Fall back to sniffing the magic bytes rather than trusting the URL.
        suffix = ".pdf" if content[:5] == b"%PDF-" else ".png"
    return content, suffix


def _extract_bytes(content: bytes, suffix: str):
    """Write to a temp file, run extraction, always clean up."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(content)
        tmp_path = tmp.name
    try:
        return extractor.extract(tmp_path)
    finally:
        os.unlink(tmp_path)


@app.get("/health")
async def health():
    from app.extractor import GEMINI_MODEL, FALLBACK_MODELS, _mock_mode

    mock = _mock_mode()
    return {
        "status": "ok",
        "version": "2.0.0",
        "model": GEMINI_MODEL,
        "fallback_models": FALLBACK_MODELS,
        # Surfaced because mock mode is silent and easy to leave on by
        # accident: an exported USE_MOCK_MODE beats the value in .env, so the
        # service can return dummy data while .env clearly says false.
        "mock_mode": mock,
        "warning": (
            "USE_MOCK_MODE is ON - all extractions return dummy data. "
            "Unset it in the shell (it overrides .env) and restart."
            if mock else None
        ),
    }



@app.post("/extract-bill-data", response_model=ExtractionResponse)
async def extract_bill_data(req: DocumentURLRequest):
    """
    Fetch a document from a URL and extract bill line items.
    """
    content, suffix = _download(req.document)
    return _extract_bytes(content, suffix)


@app.post("/extract-from-file", response_model=ExtractionResponse)
async def extract_from_file(file: UploadFile = File(...)):
    """
    Accept a multipart file upload (PDF or image) and extract bill line items.
    """
    content = await file.read()
    if len(content) > MAX_DOWNLOAD_BYTES:
        raise HTTPException(413, "File exceeds the maximum allowed size")

    suffix = Path(file.filename).suffix.lower() if file.filename else ""
    if suffix not in ALLOWED_SUFFIXES:
        suffix = ".pdf" if content[:5] == b"%PDF-" else ".png"

    return _extract_bytes(content, suffix)
