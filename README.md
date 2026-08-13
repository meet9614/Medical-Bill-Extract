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
Tesseract OSD auto-rotate    ← Correct sideways scans, 90° steps (applied to
    │                          BOTH the OCR input and the image sent to Gemini)
    ▼
OpenCV fine-angle deskew     ← The remaining few degrees of tilt from a photo,
    │                          which OSD cannot fix. Skipped on upright pages.
    ▼
Pillow enhancement + OCR     ← Contrast/sharpness, --psm 6, run in parallel
    │                          across pages (OMP_THREAD_LIMIT=1).
    │                          Engine switchable: Tesseract or PaddleOCR.
    ▼
Gemini (multimodal)          ← Page image + OCR hint → JSON mode
    │
    ├── Capped back-off, model fallback list, non-retryable errors skipped
    ├── Batch pages to reduce API calls
    │
    ▼
Aggregation                  ← Suppress Bill Summary when Detail pages exist
    │                          (identical rows are NEVER merged)
    ▼
Reconciliation               ← Compare extracted sum vs the total printed
    │                          on the bill; flag any mismatch
    ▼
ExtractionResponse JSON
```

The OCR hint is an accuracy aid, not a dependency — the model receives the page
image regardless. Set `USE_OCR_HINT=false` to skip it and save ~1–2s/page.

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

## Verified results

Measured on the supplied sample bills, not estimated.

**`train_sample_1`** — 2-page printed hospital bill, 38 line items extracted.
Reconciliation: extracted total ₹73,420.25 against the printed
`Grand Total: 73,420.25` — exact. Verified independently against the rendered
page, so the match is not the model checking its own arithmetic. Category
subtotals (`Total of PATHOLOGY`) correctly excluded from line items, and a
pathology subtotal spanning both pages handled correctly.

**`train_sample_3`** — handwritten pharmacy invoice, **rotated 90°**,
photographed on a desk. Auto-rotation corrected it; both drugs read correctly
(`Lozivate-MF ₹163`, second item at 8.9 × 30 = ₹266.94); classified
`Pharmacy Bill`; reconciled ₹429.94 against a printed ₹430.00, the 6-paisa
rounding gap absorbed by the 1% tolerance. Raised a genuine fraud flag for a
handwritten alteration over a printed amount.

**OCR configuration** — chosen by scoring against `pdftotext` ground truth:

| variant | token F1 | number F1 |
|---|---|---|
| **current** (contrast 2.0 + sharpness 2.0 + median 3, `--psm 6`) | **69.5%** | **80.7%** |
| no preprocessing, psm 6 | 58.9% | 75.1% |
| current preprocessing, psm 3 (Tesseract default) | 41.9% | 6.2% |

**OpenCV deskew** — added only after measuring it in two roles. As a *replacement*
for the Pillow pre-processing it was consistently worse, so it isn't used that way:

| variant | token F1 | number F1 |
|---|---|---|
| **Pillow (kept)** | **69.5%** | **80.7%** |
| OpenCV adaptive threshold | 66.8% | 75.1% |
| OpenCV denoise + Otsu | 58.0% | 73.3% |
| OpenCV CLAHE | 59.8% | 77.0% |

For *deskew*, which the Pillow chain cannot do at all, it wins clearly:

| page state | token F1 | number F1 |
|---|---|---|
| straight | 69.5% | 80.7% |
| tilted 2°, no deskew | 58.8% | 60.6% |
| **tilted 2°, with deskew** | **64.2%** | **73.4%** |
| tilted 4°, no deskew | 48.2% | 46.7% |
| **tilted 4°, with deskew** | **60.2%** | **72.3%** |

Upright pages detect 0° and are left untouched — rotation always costs a little
sharpness, so it's only applied when it pays for itself.

**PaddleOCR** is wired in as a switchable backend (`OCR_ENGINE=paddle`) with
automatic fallback to Tesseract. It is **not yet benchmarked** against Tesseract
on this corpus.

Note that Tesseract confidence **cannot** detect handwriting — measured means
overlap (handwritten 59.3/49.5 vs printed 65.5/86.8/88.8/74.2), so
`OCR_MIN_CONFIDENCE` is only a catastrophic-failure guard.

---

## Setup

### 1. System dependencies

```bash
# Ubuntu / Debian
sudo apt-get install poppler-utils tesseract-ocr tesseract-ocr-eng tesseract-ocr-hin

# macOS  (tesseract-lang provides the Hindi and OSD data files)
brew install poppler tesseract tesseract-lang
```

Optional — the alternative OCR engine, several hundred MB:

```bash
pip install paddlepaddle paddleocr    # then set OCR_ENGINE=paddle
```

### 2. Python dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure environment

Create `.env` in the repo root:

```
GOOGLE_API_KEY=your-key-here
GEMINI_MODEL=gemini-flash-latest
USE_MOCK_MODE=false
```

### 4. Pick a model that your key can actually use

Model availability changes, and a retired model returns a 404 that looks like an
auth failure. This script lists what your key supports **and makes a real
`generateContent` call** — listing models alone is not proof, since a restricted
project will list everything and then refuse every request:

```bash
python list_models.py
```

Copy the suggested `GEMINI_MODEL` into `.env`.

### 5. Run

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Swagger UI: <http://localhost:8000/docs>

Check the backend is configured as you expect before uploading anything:

```bash
curl -s localhost:8000/health
# {"status":"ok","model":"gemini-flash-latest","mock_mode":false,...}
```

### 6. Frontend (optional)

```bash
streamlit run app_ui.py                              # talks to localhost:8000
MEDIDATA_API=http://your-host:8000 streamlit run app_ui.py   # or a deployment
```

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Liveness check; reports active model and mock-mode state |
| POST | `/extract-bill-data` | Extract from a document URL |
| POST | `/extract-from-file` | Extract from a file upload |

Both extraction endpoints cap request size (`MAX_DOWNLOAD_BYTES`, default 50MB).
The URL endpoint resolves the hostname and rejects private, loopback and
link-local addresses — without that check, a caller could reach cloud instance
metadata or internal services through the API.

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
pytest -q      # 31 tests, mock mode, no API key, no cost
```

The suite forces `USE_MOCK_MODE=true`, so it never calls the API. Anything that
does belongs behind the registered `integration` marker.

### Measuring accuracy

Reconciliation tells you *when* an extraction is wrong; only labelled data tells
you *how* wrong. `eval/score.py` bootstraps a labelling template from current
output, which you then correct by hand:

```bash
python -m eval.score --make-template samples/train_sample_1.pdf
# edit eval/gold/train_sample_1.json, set "verified": true
python -m eval.score --score
```

It reports precision/recall/F1 on line items, matching on name **and** amount so
a model that invents plausible names with wrong numbers scores badly. Matching is
one-to-one, so a single predicted row cannot satisfy 20 identical gold rows.

It also cross-checks whether reconciliation predicts real accuracy — if F1 is
clearly higher on documents that reconcile, that signal is trustworthy on
unlabelled bills in production.

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
| `GEMINI_MODEL` | `gemini-flash-latest` | Primary model. Run `list_models.py` to verify |
| `GEMINI_MODEL_FALLBACKS` | `gemini-2.0-flash,gemini-flash-latest` | Tried in order if the primary fails |
| `MAX_RETRIES` | `5` | Retries per model before switching |
| `MAX_BACKOFF_SECONDS` | `8` | Cap on exponential back-off between retries |
| `MAX_OUTPUT_TOKENS` | `16384` | Raise if dense multi-page batches return empty |
| `BATCH_SIZE` | `3` | Pages per Gemini API call. Lower it if dense pages come back empty |
| `PDF_DPI` | `200` | DPI for PDF rendering |
| `USE_MOCK_MODE` | `false` | Return dummy data (no API calls) |
| `RECONCILE_TOLERANCE_PCT` | `1.0` | Tolerance when checking against the printed total |
| `USE_OCR_HINT` | `true` | Send the OCR text hint alongside the image |
| `OCR_MAX_CHARS` | `6000` | Max OCR hint length per page |
| `OCR_WORKERS` | `min(4, cpus)` | Parallel OCR workers (sets `OMP_THREAD_LIMIT=1`) |
| `OCR_AUTOROTATE` | `true` | Correct 90° page orientation via Tesseract OSD |
| `OCR_DESKEW` | `true` | Correct fine tilt via OpenCV (no-op on upright pages) |
| `DESKEW_MIN_ANGLE` | `0.3` | Below this, leave the page alone |
| `DESKEW_MAX_ANGLE` | `15.0` | Above this, assume detection failed |
| `OCR_ENGINE` | `tesseract` | `tesseract` or `paddle` (falls back if uninstalled) |
| `OCR_MIN_CONFIDENCE` | `35` | Drop the hint below this mean word confidence |
| `MAX_DOWNLOAD_BYTES` | `52428800` | Upload / download size cap |
| `DOWNLOAD_TIMEOUT_S` | `30` | Timeout when fetching a document URL |

---

## Differentiators

1. **Self-verifying output** — Every response reconciles the extracted line items against the total printed on the bill, so a wrong extraction is detectable per request without any labelled data.
2. **Multimodal extraction** — Page images are sent directly to Gemini so it can read tables, handwriting, stamps, and rotated text that OCR alone misses.
3. **Measured OCR pipeline** — `--psm 6` plus contrast/sharpness preprocessing, chosen by scoring against `pdftotext` ground truth: 80.7% F1 on numeric tokens versus 6.2% for Tesseract's default segmentation mode.
4. **Two-stage geometry correction** — Tesseract OSD fixes 90° rotations, OpenCV `minAreaRect` deskew fixes the residual few degrees of tilt from phone photos. Both applied before OCR *and* before the API call. Without OSD a sideways page returns pure noise; without deskew a 4° tilt costs 34 points of numeric F1.
5. **Anti-double-counting** — Bill Summary pages are suppressed when Detail pages are present, while preserving their printed totals for reconciliation.
6. **Repeated-row safety** — Identical rows are never merged. Hospital bills legitimately charge the same service 20 times, and collapsing them silently deletes money.
7. **Multilingual support** — Tesseract with Hindi (`hin`) language pack handles bilingual bills.
8. **Quota resilience** — Capped exponential back-off with a model fallback list, skipping retries for non-retryable errors.
9. **SSRF-hardened URL intake** — The URL endpoint resolves and rejects private, loopback and link-local addresses, and enforces size and time limits.

---

## Troubleshooting

Failures here are reported explicitly rather than silently — an empty result is
never returned as success. Check the `error` field first.

**`404 ... is no longer available` / `is not found for API version`**
The model name is retired or absent on your key — not an auth problem. Run
`python list_models.py` and update `GEMINI_MODEL`.

**`429 ... quota_value: 20`**
Free tier allows ~20 requests/day/model. At `BATCH_SIZE=3` that is roughly one
pass over a 50-page corpus per day, with no margin for retries or evaluation.
A full pass costs around USD 0.20 on billing; enable it if you are iterating.

**Results come back as `Mock Item 1`**
`USE_MOCK_MODE` is on. An **exported shell variable overrides `.env`**, so check
the shell too, and note `--reload` does not re-read `.env` — restart fully.
`curl localhost:8000/health` reports the effective state.

**Zero items but non-zero `output_tokens`**
The model answered and nothing parsed. The `error` field names the response
shape. Usually a dense multi-page batch: lower `BATCH_SIZE` or raise
`MAX_OUTPUT_TOKENS`.

**`ModuleNotFoundError: No module named 'app'` under pytest**
`pytest.ini` sets `pythonpath = .` — run pytest from the repo root.

**OCR is slower with more workers**
Tesseract is already internally multi-threaded; concurrent instances
oversubscribe the CPU. `extractor.py` sets `OMP_THREAD_LIMIT=1` automatically
when `OCR_WORKERS > 1`. Measured on 4 cores: 2 pages took 2.40s serial, 17.42s
with 2 workers unrestricted, and 2.1× faster than serial once the limit is set.

---

## Project Structure

```
.
├── app/
│   ├── main.py         # FastAPI routes, URL intake + SSRF guards
│   ├── extractor.py    # Core pipeline (rotate → OCR → Gemini → reconcile)
│   └── schemas.py      # Pydantic models
├── eval/
│   └── score.py        # Label templates + accuracy scoring (stdlib only)
├── tests/
│   └── test_api.py     # 31 unit + API tests, all in mock mode
├── app_ui.py           # Streamlit frontend
├── list_models.py      # Which models can this key actually call?
├── DATA_NOTES.md       # What is actually in the sample bills — read this
├── Dockerfile
├── pytest.ini
└── requirements.txt
```

[`DATA_NOTES.md`](DATA_NOTES.md) documents the sample corpus: 12 of 15 files are
scanned images, two "samples" are slices of the same 90-page bill, repeated line
items are legitimate and must never be deduplicated, and the anonymisation
white-boxes look exactly like the fraud signal being prompted for.
