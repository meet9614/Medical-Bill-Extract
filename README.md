# Medical Bill Extractor

Production-ready FastAPI service that extracts structured line items from hospital bills and invoices using a **hybrid OCR + Gemini multimodal pipeline**.

Built for the **Bajaj Finserv Health Datathon 2025**.

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
    "grand_total": 73420.25
  },
  "error": null
}
```

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

Frontend: https://medical-bill-extract-2ashatbzcxn9fkhejaxboe.streamlit.app/

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
| `BATCH_SIZE` | `3` | Pages per Gemini API call |
| `PDF_DPI` | `200` | DPI for PDF rendering |
| `USE_MOCK_MODE` | `false` | Return dummy data (no API calls) |

---

## Differentiators

1. **Multimodal extraction** — Page images are sent directly to Gemini so it can read tables, handwriting, stamps, and rotated text that OCR alone misses.
2. **OCR hint** — Tesseract output is passed alongside the image as a text hint, improving accuracy on printed text.
3. **Fraud detection** — Gemini is prompted to flag mismatched fonts, white-out over text, and rate × quantity mismatches.
4. **Anti-double-counting** — Bill Summary pages are automatically suppressed when Detail pages are present.
5. **Multilingual support** — Tesseract with Hindi (`hin`) language pack handles bilingual bills.
6. **Quota resilience** — Automatic retry with exponential back-off + model fallback list.
7. **Pre-processing** — Contrast enhancement and median filtering improve OCR quality on low-quality scans.

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
