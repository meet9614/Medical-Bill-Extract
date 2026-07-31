# MediData: A Medical Invoice Analyser

Production-ready FastAPI service that extracts structured line items from hospital bills and invoices using a **hybrid OCR + Gemini multimodal pipeline**.

Built for the **Byterverse Hackathon 2026**.

---

## Architecture

```
PDF / Image
    │
    ▼
pdf2image (poppler)          ← Render each page at configurable DPI
    │
    ▼
Pillow image enhancement     ← Contrast / sharpness boost
    │
    ├──▶ pytesseract OCR     ← Raw text hint (handles printed text well)
    │
    ▼
Gemini 2.5 Flash (multimodal) ← Page image + OCR hint → structured JSON
    │
    ├── Retry + fallback model list (quota resilience)
    ├── Batch pages to reduce API calls
    │
    ▼
Deduplication logic          ← Suppress Bill Summary when Detail pages exist
    │
    ▼
ExtractionResponse JSON
```

---

## Response Schema

```json
{
  "is_success": true,
  "token_usage": {
    "total_tokens": 1234,
    "input_tokens": 1000,
    "output_tokens": 234
  },
  "data": {
    "pagewise_line_items": [
      {
        "page_no": "1",
        "page_type": "Bill Detail",
        "bill_items": [
          {
            "item_name": "BED CHARGE GENERAL WARD",
            "item_amount": 1500.00,
            "item_rate": 1500.00,
            "item_quantity": 1.0
          }
        ]
      }
    ],
    "total_item_count": 42,
    "grand_total": 73420.25,
    "reconciliation": {
      "printed_total": 73420.25,
      "computed_total": 73420.25,
      "difference": 0.0,
      "pct_difference": 0.0,
      "matches": true,
      "note": "Extracted items match the printed total."
    },
    "fraud_flags": []
  },
  "error": null
}
```

### Reconciliation

Every bill states its own grand total, so the extractor copies that printed
figure and compares it against the sum of the line items it extracted. This is a
correctness check that needs no labelled data:

| `matches` | meaning |
|---|---|
| `true` | extracted items sum to the printed total (within 1%) |
| `false` | **provably wrong** — rows were missed, or something was double-counted |
| `null` | the document printed no total, so nothing to verify against |

A negative `difference` means items were missed or the upload was a partial
excerpt; a positive one usually means a summary page was counted alongside
detail pages. Use it as a per-request confidence signal.

`page_type` values: `Bill Summary` | `Bill Detail` | `Pharmacy Bill` | `Lab Bill` | `Other`

---

## Setup

### 1. System dependencies

```bash
# Ubuntu / Debian
sudo apt-get install poppler-utils tesseract-ocr tesseract-ocr-eng tesseract-ocr-hin

# macOS
brew install poppler tesseract
```

### 2. Python dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure environment

```bash
cp .env.example .env
# Edit .env and set GOOGLE_API_KEY
```

### 4. Run

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Swagger UI: [http://localhost:8000/docs](http://13.206.108.88:8000/docs)

Frontend: [https://medical-bill-extract-2ashatbzcxn9fkhejaxboe.streamlit.app/](https://medical-bill-extract-2ashatbzcxn9fkhejaxboe.streamlit.app/)

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Liveness check |
| POST | `/extract-from-file` | Extract from file upload |

### Extract from URL

```bash
curl -X POST http://localhost:8000/extract-bill-data \
  -H "Content-Type: application/json" \
  -d '{"document": "https://example.com/bill.pdf"}'
```

### Extract from file

```bash
curl -X POST http://localhost:8000/extract-from-file \
  -F "file=@/path/to/bill.pdf"
```

---

## Testing

```bash
# Fast (no API key needed – mock mode)
pytest -q

# With real API
USE_MOCK_MODE=false pytest -q -m integration
```

---

## Docker

```bash
docker build -t medical-bill-extractor .
docker run -p 8000:8000 --env-file .env medical-bill-extractor
```

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `GOOGLE_API_KEY` | — | **Required.** Google AI API key |
| `GEMINI_MODEL` | `gemini-2.5-flash` | Primary model |
| `GEMINI_MODEL_FALLBACKS` | `gemini-2.0-flash,gemini-1.5-flash` | Fallback models |
| `MAX_RETRIES` | `5` | Retries per model before switching |
| `MAX_BACKOFF_SECONDS` | `8` | Cap on exponential back-off between retries |
| `BATCH_SIZE` | `3` | Pages per Gemini API call |
| `PDF_DPI` | `200` | DPI for PDF rendering |
| `USE_MOCK_MODE` | `false` | Return dummy data (no API calls) |
| `RECONCILE_TOLERANCE_PCT` | `1.0` | Tolerance when checking against the printed total |
| `USE_OCR_HINT` | `true` | Send the OCR text hint alongside the image |
| `OCR_MAX_CHARS` | `6000` | Max OCR hint length per page |
| `OCR_WORKERS` | `min(4, cpus)` | Parallel OCR workers (sets `OMP_THREAD_LIMIT=1`) |
| `OCR_AUTOROTATE` | `true` | Correct page orientation via Tesseract OSD |
| `OCR_MIN_CONFIDENCE` | `35` | Drop the hint below this mean word confidence |
| `MAX_DOWNLOAD_BYTES` | `52428800` | Upload / download size cap |
| `DOWNLOAD_TIMEOUT_S` | `30` | Timeout when fetching a document URL |

---

## Differentiators

1. **Self-verifying output** — Every response reconciles the extracted line items against the total printed on the bill, so a wrong extraction is detectable per request without any labelled data.
2. **Multimodal extraction** — Page images are sent directly to Gemini so it can read tables, handwriting, stamps, and rotated text that OCR alone misses.
3. **Measured OCR pipeline** — `--psm 6` plus contrast/sharpness preprocessing, chosen by scoring against `pdftotext` ground truth: 80.7% F1 on numeric tokens versus 6.2% for Tesseract's default segmentation mode.
4. **Auto-rotation** — Tesseract OSD corrects sideways pages before both OCR and the API call; without it a 90°-rotated page returns noise.
5. **Anti-double-counting** — Bill Summary pages are suppressed when Detail pages are present, while preserving their printed totals for reconciliation.
6. **Repeated-row safety** — Identical rows are never merged. Hospital bills legitimately charge the same service 20 times, and collapsing them silently deletes money.
7. **Multilingual support** — Tesseract with Hindi (`hin`) language pack handles bilingual bills.
8. **Quota resilience** — Capped exponential back-off with a model fallback list, skipping retries for non-retryable errors.
9. **SSRF-hardened URL intake** — The URL endpoint resolves and rejects private, loopback and link-local addresses, and enforces size and time limits.

---

## Project Structure

```
.
├── app/
│   ├── main.py        # FastAPI routes
│   ├── extractor.py   # Core pipeline (OCR + Gemini + aggregation)
│   └── schemas.py     # Pydantic models
├── tests/
│   └── test_api.py    # Unit + API tests
├── Dockerfile
├── render.yaml
├── requirements.txt
├── pytest.ini
└── .env.example
```
