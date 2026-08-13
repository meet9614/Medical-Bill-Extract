"""
Medical Bill Extractor - Core Extraction Pipeline

Architecture:
  1. PDF → Images (pdf2image / pillow)
  2. Per-page OCR (pytesseract) → raw text hint
  3. Batch pages (configurable) → send images + OCR text to Gemini multimodal
  4. Gemini returns structured JSON per page
  5. Aggregate pages, deduplicate, compute totals
  6. Return ExtractionResponse

Key design decisions:
  - Multimodal: Send the actual page *image* to Gemini so it can see layout,
    tables, handwriting, stamps, etc. that OCR alone would miss.
  - OCR text is passed as a text hint alongside the image for better accuracy.
  - Fraud detection: Gemini is prompted to flag inconsistent fonts / whitener.
  - Retry + fallback model list for quota resilience.
  - Deduplication: If a page appears on both a Summary and a Detail page,
    we keep only Detail-page items so we don't double-count.
"""

import os
import re
import json
import time
import base64
import logging
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List, Optional, Tuple

from dotenv import load_dotenv


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENV_PATH = os.path.join(BASE_DIR, ".env")

load_dotenv(dotenv_path=ENV_PATH)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# ── Optional heavy dependencies (graceful fallback) ────────────────────────
try:
    from pdf2image import convert_from_path
    PDF2IMAGE_AVAILABLE = True
except ImportError:
    PDF2IMAGE_AVAILABLE = False
    logger.warning("pdf2image not available – PDF page rendering disabled")

try:
    import pytesseract
    from PIL import Image, ImageEnhance, ImageFilter
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False
    logger.warning("pytesseract / Pillow not available – OCR disabled")

try:
    import cv2
    import numpy as np
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    logger.warning("opencv not available – deskew disabled, pages will be OCR'd as-is")

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    logger.warning("google-generativeai not available – LLM extraction disabled")

from app.schemas import (
    BillItem,
    ExtractionData,
    ExtractionResponse,
    PageLineItems,
    Reconciliation,
    TokenUsage,
)

# ── Config ─────────────────────────────────────────────────────────────────
GOOGLE_API_KEY   = os.getenv("GOOGLE_API_KEY", "")
# Never log any part of the key. Prefixes are enough to identify a key in a
# leaked log, and server logs routinely end up in shared storage.
logger.info(
    "config loaded from %s | GOOGLE_API_KEY %s",
    ENV_PATH,
    "present" if GOOGLE_API_KEY else "MISSING",
)
# Default is the rolling alias, not a pinned version: gemini-1.5-flash and then
# gemini-2.5-flash both went stale in this repo and 404'd for new keys. Pin an
# explicit version in .env for reproducibility; the alias is only a safe default.
GEMINI_MODEL     = os.getenv("GEMINI_MODEL", "gemini-flash-latest")
# gemini-1.5-flash was retired and now 404s -- keeping it here only wasted a
# fallback slot and buried the real error behind a misleading one. Run
# `python list_models.py` to see what a given key can actually call.
FALLBACK_MODELS  = [
    m.strip()
    for m in os.getenv(
        "GEMINI_MODEL_FALLBACKS", "gemini-2.0-flash,gemini-flash-latest"
    ).split(",")
    if m.strip()
]
MAX_RETRIES      = int(os.getenv("MAX_RETRIES", "5"))
MAX_BACKOFF_SECONDS = int(os.getenv("MAX_BACKOFF_SECONDS", "8"))
# A dense bill page can carry 25+ line items; three of them in one batch can
# exceed the old 8192 default, and a truncated response parses to nothing.
MAX_OUTPUT_TOKENS = int(os.getenv("MAX_OUTPUT_TOKENS", "16384"))
# Percentage tolerance when checking extracted items against the printed total.
RECONCILE_TOLERANCE_PCT = float(os.getenv("RECONCILE_TOLERANCE_PCT", "1.0"))
BATCH_SIZE       = int(os.getenv("BATCH_SIZE", "3"))
PDF_DPI          = int(os.getenv("PDF_DPI", "200"))
def _mock_mode() -> bool:
    """
    Read at call time, not import time.

    Module-level constants are frozen when uvicorn first imports the app, and
    `--reload` only watches .py files -- so editing USE_MOCK_MODE in .env has no
    effect until the server is fully restarted. Reading it per call means an
    exported env var works immediately, and a .env edit works after any code
    reload rather than requiring a restart.
    """
    return os.getenv("USE_MOCK_MODE", "false").strip().lower() == "true"

# ── OCR tuning ─────────────────────────────────────────────────────────────
# Was hard-coded to 1500, which discarded 27-33% of the OCR text on dense bill
# pages -- and truncated from the END, exactly where the later line items and
# totals live. A 6000-char hint costs ~1500 input tokens (~$0.0002/page), so
# there is no reason to cut it this close.
OCR_MAX_CHARS = int(os.getenv("OCR_MAX_CHARS", "6000"))

# Page-level parallelism. pytesseract shells out to the tesseract binary, so
# the GIL is released and threads do work here.
#
# CRITICAL: tesseract 4.x is already internally multi-threaded via OpenMP, so N
# concurrent instances each spawn their own pool and oversubscribe the CPU.
# Measured on a 4-core box, 2 pages:
#
#     serial                          2.40s
#     2 workers, OpenMP unrestricted 17.42s   <- 7x SLOWER
#     serial,    OMP_THREAD_LIMIT=1   7.49s
#     3 workers, OMP_THREAD_LIMIT=1   3.57s   <- 2.1x faster
#
# So the env var below is not a micro-optimisation; without it the thread pool
# is a large regression. It must be set before the first tesseract call, hence
# module scope.
OCR_WORKERS = int(os.getenv("OCR_WORKERS", str(min(4, (os.cpu_count() or 1)))))
if OCR_WORKERS > 1:
    os.environ.setdefault("OMP_THREAD_LIMIT", "1")

# Detect and correct page rotation before OCR *and* before sending the image to
# the model. Measured: a 90-degree-rotated page yields pure noise without this.
OCR_AUTOROTATE = os.getenv("OCR_AUTOROTATE", "true").lower() == "true"

# Fine-angle deskew with OpenCV. Complements OSD rather than replacing it:
# Tesseract's OSD only corrects in 90-degree steps, so a page photographed a few
# degrees off stays crooked, and Tesseract's line finder degrades badly on tilt.
#
# Measured on a page with pdftotext ground truth, numeric-token F1:
#     straight page                       80.7%
#     tilted 2 deg, no deskew             60.6%   -> with deskew  73.4%
#     tilted 4 deg, no deskew             46.7%   -> with deskew  72.3%
#
# Skipped entirely when the detected angle is under DESKEW_MIN_ANGLE, so upright
# scans are never resampled (rotation always costs a little sharpness).
OCR_DESKEW = os.getenv("OCR_DESKEW", "true").lower() == "true"
DESKEW_MIN_ANGLE = float(os.getenv("DESKEW_MIN_ANGLE", "0.3"))
DESKEW_MAX_ANGLE = float(os.getenv("DESKEW_MAX_ANGLE", "15.0"))

# Which OCR engine supplies the text hint.
#   tesseract - default, always available
#   paddle    - PaddleOCR; better text detection on skewed/photographed pages,
#               but a several-hundred-MB dependency. Install with:
#                   pip install paddlepaddle paddleocr
# Falls back to tesseract automatically if paddle is selected but not installed.
OCR_ENGINE = os.getenv("OCR_ENGINE", "tesseract").strip().lower()

# Mean word confidence below which the OCR hint is dropped entirely.
#
# This is a CATASTROPHIC-GARBAGE GUARD, not a handwriting detector. Measured
# mean confidence on this corpus:
#
#     handwritten, rotated   59.3      printed, dense    65.5
#     handwritten, on cloth  49.5      printed, dense    86.8
#                                      printed, clean    88.8 / 74.2
#
# The distributions overlap: a dense printed page (65.5) scores higher than one
# handwritten page and lower than the other. No threshold separates them, and
# median confidence and %low-confidence-words do no better. So tesseract's
# confidence CANNOT be used to detect handwriting, and a threshold tuned to
# catch the handwritten pages would also silently discard good printed ones.
#
# 35 therefore only catches pages where OCR has collapsed completely. Do not
# raise it to try to catch handwriting -- the measurements above show it would
# discard good printed pages at the same time.
OCR_MIN_CONFIDENCE = float(os.getenv("OCR_MIN_CONFIDENCE", "35"))

# The model receives the page image regardless, so the OCR hint is an optional
# accuracy aid, not a dependency. Set USE_OCR_HINT=false to skip it and save
# ~1-2s/page; compare extracted totals with it on and off to decide whether it
# earns that latency on your documents.
USE_OCR_HINT = os.getenv("USE_OCR_HINT", "true").lower() == "true"

# ── System prompt for Gemini ───────────────────────────────────────────────
SYSTEM_PROMPT = """
You are an expert medical billing analyst. Your task is to extract ALL line items
from hospital / pharmacy / diagnostic bills (invoices).

For each PAGE provided, return a JSON object with the following structure:
{
  "page_no": "<integer as string, 1-indexed>",
  "page_type": "<one of: Bill Summary | Bill Detail | Pharmacy Bill | Lab Bill | Other>",
  "bill_items": [
    {
      "item_name": "<description of service / drug / test>",
      "item_amount": <float, total amount for this line>,
      "item_rate": <float or null, unit rate>,
      "item_quantity": <float or null, quantity>
    }
  ],
  "printed_total": <float or null, the total PRINTED on this page>,
  "fraud_flags": ["<optional list of suspicious observations>"]
}

IMPORTANT RULES:
1. Extract EVERY line item – do not skip any row in any table.
2. Do NOT include sub-totals or grand-totals as line items.
3. If the page is a SUMMARY page (category totals only, no individual items),
   set page_type = "Bill Summary" and bill_items = [].
   The Summary totals will NOT be added to the final total.
4. If item_rate and item_quantity are not explicitly given, derive them if possible;
   otherwise set to null.
5. Amounts must be numeric floats, not strings. Strip currency symbols (₹, Rs., $).
6. For pharmacy memos: each drug line is one item_name. Use the printed Amount column.
7. If a line shows a returned / credit item (negative), include it with a negative amount.
8. IDENTICAL REPEATED ROWS ARE REAL. Hospital bills legitimately charge the same
   service many times (e.g. 20 rows of "BLOOD SUGAR BY GLUCOMETER", or the same
   consultation fee on consecutive days). Emit every occurrence as its own item.
   NEVER collapse, merge or deduplicate repeated rows.
9. printed_total: the grand total / net payable AS PRINTED on the page, if one
   appears. Copy the printed figure exactly – do not compute it yourself, and do
   not infer one if the page shows none (use null). This is used to verify the
   extraction, so a computed value would defeat the check.
10. Fraud indicators to flag: amounts that do not match rate × quantity,
    duplicate serial numbers, handwritten alterations over printed text,
    totals that disagree with the sum of the lines.
    Do NOT flag white boxes or blanked-out regions – documents in this pipeline
    are anonymised by covering patient names and letterheads, and that redaction
    is expected, not suspicious.
11. Return ONLY valid JSON – no markdown fences, no prose.
"""

# ── Helper ─────────────────────────────────────────────────────────────────

def _encode_image(path: str) -> str:
    """Base64-encode an image file."""
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def _enhance_image(img) -> "Image":
    """Improve contrast / sharpness before OCR."""
    img = img.convert("L")  # greyscale
    img = ImageEnhance.Contrast(img).enhance(2.0)
    img = ImageEnhance.Sharpness(img).enhance(2.0)
    img = img.filter(ImageFilter.MedianFilter(size=3))
    return img


def _autorotate(img):
    """
    Correct page orientation using Tesseract's OSD.

    Returns (possibly rotated image, degrees applied). Applied to the page
    before OCR *and* before the image is sent to the model -- a sideways page
    hurts both. OSD's own confidence is often low even when the answer is
    right, so we act on any non-zero rotation and log the confidence rather
    than gating on it.
    """
    if not (OCR_AVAILABLE and OCR_AUTOROTATE):
        return img, 0
    try:
        osd = pytesseract.image_to_osd(img, output_type=pytesseract.Output.DICT)
        rotate = int(osd.get("rotate", 0)) % 360
        if rotate:
            logger.info(
                "page rotated %d deg (osd confidence %.1f)",
                rotate, float(osd.get("orientation_conf", 0.0)),
            )
            # PIL rotates counter-clockwise; OSD's `rotate` is the clockwise
            # correction needed, so expand=True and negate.
            return img.rotate(-rotate, expand=True), rotate
    except Exception as e:  # noqa: BLE001
        logger.debug(f"OSD unavailable for this page: {e}")
    return img, 0


def _deskew(img):
    """
    Correct a small page tilt using OpenCV. Returns (image, degrees applied).

    How it works: threshold the page, take the coordinates of every dark pixel,
    and ask cv2.minAreaRect for the tightest rotated rectangle around them. On a
    document that rectangle aligns with the text baselines, so its angle is the
    page tilt. We then rotate by the NEGATIVE of that angle to flatten it.

    The sign matters and is easy to get wrong -- rotating the same direction as
    the detected angle doubles the tilt and made OCR markedly worse in testing.
    """
    if not (CV2_AVAILABLE and OCR_DESKEW):
        return img, 0.0
    try:
        grey = np.array(img.convert("L"))
        inverted = cv2.bitwise_not(grey)
        mask = cv2.threshold(inverted, 0, 255,
                             cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

        coords = np.column_stack(np.where(mask > 0))
        if coords.shape[0] < 100:          # nearly blank page, nothing to align
            return img, 0.0

        angle = cv2.minAreaRect(coords)[-1]
        # minAreaRect reports within (-90, 0]; normalise to a small signed tilt
        if angle < -45:
            angle = 90 + angle
        elif angle > 45:
            angle -= 90

        # Ignore noise, and refuse implausible angles -- a genuine 90-degree
        # rotation is OSD's job, and a wild value here means detection failed.
        if abs(angle) < DESKEW_MIN_ANGLE or abs(angle) > DESKEW_MAX_ANGLE:
            return img, 0.0

        height, width = grey.shape
        matrix = cv2.getRotationMatrix2D((width // 2, height // 2), -angle, 1.0)
        rotated = cv2.warpAffine(
            np.array(img.convert("L")), matrix, (width, height),
            flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE,
        )
        logger.info("deskewed page by %.2f degrees", -angle)
        return Image.fromarray(rotated), -angle
    except Exception as e:  # noqa: BLE001
        logger.debug(f"deskew failed, using page as-is: {e}")
        return img, 0.0


_paddle_reader = None


def _ocr_page_paddle(img) -> str:
    """
    OCR via PaddleOCR. Optional alternative to Tesseract.

    Worth trying because Paddle's detection stage handles skewed and
    photographed text better than Tesseract's, which matters on phone-camera
    bills. Costs a several-hundred-MB dependency, so it is off by default.

    Returns "" on any failure, which makes the caller fall back to Tesseract.
    """
    global _paddle_reader
    try:
        if _paddle_reader is None:
            from paddleocr import PaddleOCR
            _paddle_reader = PaddleOCR(use_angle_cls=True, lang="en", show_log=False)
            logger.info("PaddleOCR initialised")

        result = _paddle_reader.ocr(np.array(img.convert("RGB")), cls=True)
        if not result or not result[0]:
            return ""

        # Paddle returns [[box, (text, confidence)], ...] per page.
        lines = []
        for entry in result[0]:
            try:
                text, confidence = entry[1]
            except (IndexError, TypeError, ValueError):
                continue
            if float(confidence) >= 0.5:
                lines.append(str(text))
        return "\n".join(lines).strip()

    except ImportError:
        logger.warning("OCR_ENGINE=paddle but paddleocr is not installed "
                       "(pip install paddlepaddle paddleocr) - using tesseract")
        return ""
    except Exception as e:  # noqa: BLE001
        logger.warning(f"PaddleOCR failed, falling back to tesseract: {e}")
        return ""


def _ocr_page(img) -> str:
    """
    OCR a PIL image, returning text only if it looks trustworthy.

    Uses image_to_data rather than image_to_string so text and per-word
    confidence come from a SINGLE tesseract pass -- calling both would double
    the cost of the slowest step in the pipeline.

    --psm 6 is deliberate and measured: against pdftotext ground truth it
    scored 80.7% F1 on numeric tokens versus 6.2% for the psm 3 default. Do not
    "fix" it to auto page segmentation.
    """
    # Optional alternative engine. If it returns nothing (not installed, or it
    # failed) we silently continue with Tesseract rather than losing the hint.
    if OCR_ENGINE == "paddle":
        text = _ocr_page_paddle(img)
        if text:
            return text[:OCR_MAX_CHARS]

    if not OCR_AVAILABLE:
        return ""
    try:
        enhanced = _enhance_image(img.copy())
        data = pytesseract.image_to_data(
            enhanced, config="--psm 6", output_type=pytesseract.Output.DICT
        )

        lines: dict[tuple, list] = {}
        confs = []
        for i, word in enumerate(data["text"]):
            if not word.strip():
                continue
            try:
                conf = float(data["conf"][i])
            except (TypeError, ValueError):
                continue
            if conf < 0:
                continue
            key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
            lines.setdefault(key, []).append(word)
            confs.append(conf)

        if not confs:
            return ""

        mean_conf = sum(confs) / len(confs)
        if mean_conf < OCR_MIN_CONFIDENCE:
            logger.info(
                "dropping OCR hint: mean confidence %.1f < %.1f (OCR has "
                "collapsed on this page; the model still gets the image)",
                mean_conf, OCR_MIN_CONFIDENCE,
            )
            return ""
        if mean_conf < 60:
            # Not actionable automatically -- see the note on OCR_MIN_CONFIDENCE
            # for why confidence can't be thresholded into a handwriting
            # detector -- but useful when auditing which pages went wrong.
            logger.debug("low-confidence OCR (%.1f) on this page", mean_conf)

        text = "\n".join(" ".join(ws) for _, ws in sorted(lines.items()))
        return text.strip()
    except Exception as e:  # noqa: BLE001
        logger.warning(f"OCR failed: {e}")
        return ""


def _pdf_to_images(pdf_path: str) -> List:
    """Convert PDF pages to PIL Image objects."""
    if PDF2IMAGE_AVAILABLE:
        try:
            return convert_from_path(pdf_path, dpi=PDF_DPI)
        except Exception as e:
            logger.warning(f"pdf2image failed: {e}")
    # Fallback: try pypdf text extraction
    return []


def _clean_json(raw: str) -> str:
    """Strip markdown fences and leading/trailing whitespace."""
    raw = raw.strip()
    raw = re.sub(r"^```(?:json)?", "", raw, flags=re.MULTILINE)
    raw = re.sub(r"```$", "", raw, flags=re.MULTILINE)
    return raw.strip()


def _parse_page_result(raw_json: str, page_no: int) -> Tuple[PageLineItems, list]:
    """Parse Gemini JSON output for a single page."""
    try:
        data = json.loads(_clean_json(raw_json))
    except json.JSONDecodeError as e:
        logger.error(f"JSON parse error on page {page_no}: {e}\nRaw: {raw_json[:300]}")
        return PageLineItems(page_no=str(page_no), page_type="Other", bill_items=[]), []

    # A page that parses as valid JSON but yields no items is the silent
    # failure mode: tokens were spent, the response looked fine, and the caller
    # gets an empty bill. Log the shape so the mismatch is visible.
    if not data.get("bill_items"):
        logger.warning(
            "page %s parsed OK but contains NO bill_items. Top-level keys: %s | raw: %s",
            page_no, list(data.keys()), raw_json[:400].replace("\n", " "),
        )

    items = []
    for raw_item in data.get("bill_items", []):
        try:
            items.append(
                BillItem(
                    item_name=str(raw_item.get("item_name", "Unknown")),
                    item_amount=float(raw_item.get("item_amount", 0) or 0),
                    item_rate=_safe_float(raw_item.get("item_rate")),
                    item_quantity=_safe_float(raw_item.get("item_quantity")),
                )
            )
        except Exception:
            pass

    fraud_flags = [str(f) for f in (data.get("fraud_flags") or []) if str(f).strip()]

    page_line = PageLineItems(
        page_no=str(data.get("page_no", page_no)),
        page_type=str(data.get("page_type", "Bill Detail")),
        bill_items=items,
        printed_total=_safe_float(data.get("printed_total")),
        fraud_flags=fraud_flags,
    )
    return page_line, fraud_flags


def _safe_float(val) -> Optional[float]:
    if val is None:
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


# ── Gemini caller ──────────────────────────────────────────────────────────

class GeminiCaller:
    def __init__(self):
        if GEMINI_AVAILABLE and GOOGLE_API_KEY:
            genai.configure(api_key=GOOGLE_API_KEY)
        self.model_queue = [GEMINI_MODEL] + FALLBACK_MODELS
        self._input_tokens = 0
        self._output_tokens = 0
        # Failure bookkeeping. Without this, a batch where every model failed
        # returns "{}" per page, which parses cleanly into zero line items --
        # and the caller cannot distinguish "this bill has no items" from
        # "the API never answered".
        self.calls_attempted = 0
        self.calls_failed = 0
        self.last_error: Optional[str] = None

    @property
    def token_usage(self) -> TokenUsage:
        total = self._input_tokens + self._output_tokens
        return TokenUsage(
            total_tokens=total,
            input_tokens=self._input_tokens,
            output_tokens=self._output_tokens,
        )

    def call(self, page_images: List[str], ocr_hints: List[str], page_numbers: List[int]) -> List[str]:
        """
        Call Gemini with a batch of page images.
        Returns a list of raw JSON strings (one per page).
        """
        if _mock_mode():
            logger.info("USE_MOCK_MODE is on -- returning dummy data, no API call")
            return [self._mock_response(i) for i in range(len(page_images))]

        self.calls_attempted += 1

        if not GEMINI_AVAILABLE or not GOOGLE_API_KEY:
            logger.error("Gemini not configured – set GOOGLE_API_KEY")
            self.calls_failed += 1
            self.last_error = (
                "Gemini is not configured: GOOGLE_API_KEY is missing or the "
                "google-generativeai package is not installed."
            )
            return ["{}"] * len(page_images)

        # Build content parts: system prompt + pages
        parts = [{"text": SYSTEM_PROMPT}]

        for idx, (img_b64, ocr_text, pg_no) in enumerate(
            zip(page_images, ocr_hints, page_numbers)
        ):
            hint = (ocr_text or "")[:OCR_MAX_CHARS] if USE_OCR_HINT else ""
            parts.append(
                {
                    "text": (
                        f"\n\n--- PAGE {pg_no} ---\n"
                        f"OCR HINT:\n{hint if hint else '(no reliable OCR - read the image directly)'}\n"
                        f"\nReturn JSON for this page only:\n"
                    )
                }
            )
            parts.append(
                {
                    "inline_data": {
                        "mime_type": "image/jpeg",
                        "data": img_b64,
                    }
                }
            )

        parts.append(
            {
                "text": (
                    f"\nReturn a JSON ARRAY with exactly {len(page_images)} objects "
                    "(one per page, in order). No markdown, no prose."
                )
            }
        )

        last_error = None
        # Record why EACH model failed, not just the last one. Reporting only
        # the final fallback's error hides the primary model's failure, which is
        # nearly always the one that explains the problem.
        per_model_errors: dict = {}
        for model_name in self.model_queue:
            for attempt in range(MAX_RETRIES):
                try:
                    model = genai.GenerativeModel(model_name)

                    # Ask for JSON natively. Without this the model wraps output
                    # in markdown fences or prefaces it with prose, and we are
                    # left regex-stripping it downstream. Older SDK versions
                    # reject the argument, so fall back rather than hard-fail.
                    try:
                        cfg = genai.types.GenerationConfig(
                            # temperature=0 for reproducibility. At 0.1 the same
                            # bill produced "AL Extin" @ 8.898 on one run and
                            # "ALE-Eats" @ 8.9 on the next -- harmless here, but
                            # it makes evaluation runs non-comparable.
                            temperature=0.0,
                            max_output_tokens=MAX_OUTPUT_TOKENS,
                            response_mime_type="application/json",
                        )
                    except TypeError:
                        logger.debug("SDK lacks response_mime_type; using plain config")
                        cfg = genai.types.GenerationConfig(
                            temperature=0.0, max_output_tokens=MAX_OUTPUT_TOKENS
                        )

                    response = model.generate_content(parts, generation_config=cfg)
                    raw = response.text

                    # Track token usage if available
                    try:
                        usage = response.usage_metadata
                        self._input_tokens += usage.prompt_token_count or 0
                        self._output_tokens += usage.candidates_token_count or 0
                    except Exception:
                        pass

                    results = self._split_batch_response(raw, len(page_images))
                    return results

                except Exception as e:
                    last_error = e
                    per_model_errors[model_name] = f"{type(e).__name__}: {e}"
                    # Retrying a malformed request or a bad key fails identically
                    # every time. Only back off for transient conditions --
                    # quota, rate limits, timeouts, 5xx.
                    msg = str(e).lower()
                    transient = any(
                        k in msg
                        for k in (
                            "quota", "rate limit", "429", "timeout", "deadline",
                            "unavailable", "503", "500", "internal", "overloaded",
                            "connection", "temporarily",
                        )
                    )
                    if not transient:
                        logger.error(
                            f"Gemini {model_name}: non-retryable error, moving to "
                            f"next model: {e}"
                        )
                        break

                    # Capped at 8s. Uncapped 2**attempt over 5 retries x 3 models
                    # is ~93s of sleeping per batch, which on a 90-page bill is
                    # minutes of dead time before anything surfaces.
                    wait = min(2 ** attempt, MAX_BACKOFF_SECONDS)
                    logger.warning(
                        f"Gemini {model_name} attempt {attempt+1}/{MAX_RETRIES} "
                        f"failed: {e}. Retrying in {wait}s"
                    )
                    time.sleep(wait)

        self.calls_failed += 1
        self.last_error = " | ".join(
            f"{m}: {err}" for m, err in per_model_errors.items()
        ) or f"{type(last_error).__name__}: {last_error}"

        logger.error("All Gemini models failed for this batch:")
        for m, err in per_model_errors.items():
            logger.error("  %-24s %s", m, err)

        if any("not found" in e.lower() or "404" in e for e in per_model_errors.values()):
            logger.error(
                "  -> A 404 here means the model name does not exist on this API "
                "key/version, not that the key is invalid. List what you can "
                "actually use:\n"
                "     python -c \"import os,google.generativeai as g; "
                "from dotenv import load_dotenv; load_dotenv(); "
                "g.configure(api_key=os.getenv('GOOGLE_API_KEY')); "
                "print([m.name for m in g.list_models() "
                "if 'generateContent' in m.supported_generation_methods])\"\n"
                "     Then set GEMINI_MODEL and GEMINI_MODEL_FALLBACKS in .env."
            )
        return ["{}"] * len(page_images)

    def _split_batch_response(self, raw: str, expected: int) -> List[str]:
        """
        Gemini returns a JSON array when multiple pages are in one batch.
        Split into individual JSON strings.
        """
        raw = _clean_json(raw)
        # Try parsing as array first
        try:
            arr = json.loads(raw)
            if isinstance(arr, list):
                if len(arr) != expected:
                    logger.warning(
                        "batch split: model returned %d objects, expected %d "
                        "pages -- pages will be misaligned",
                        len(arr), expected,
                    )
                return [json.dumps(item) for item in arr]
            if isinstance(arr, dict):
                # JSON mode makes models prone to wrapping the array in an
                # object ({"pages": [...]}). Unwrap a single list-valued key
                # rather than treating the whole envelope as one page.
                list_values = [v for v in arr.values() if isinstance(v, list)]
                if expected > 1 and len(list_values) == 1 and len(list_values[0]) == expected:
                    logger.info(
                        "batch split: unwrapped %d pages from envelope key(s) %s",
                        expected, list(arr.keys()),
                    )
                    return [json.dumps(item) for item in list_values[0]]
                if expected > 1:
                    logger.warning(
                        "batch split: expected %d pages but got a single object "
                        "with keys %s -- %d page(s) will be empty",
                        expected, list(arr.keys()), expected - 1,
                    )
                return [json.dumps(arr)] + ["{}"] * (expected - 1)
        except json.JSONDecodeError as e:
            logger.warning("batch response is not valid JSON (%s); falling back "
                           "to regex object extraction. First 300 chars: %s",
                           e, raw[:300].replace("\n", " "))

        # Fallback: try to find individual JSON objects
        objects = re.findall(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)?\}", raw, re.DOTALL)
        if objects:
            return objects[:expected] + ["{}"] * max(0, expected - len(objects))

        return ["{}"] * expected

    @staticmethod
    def _mock_response(idx: int) -> str:
        """Return a mock response for testing without API key."""
        return json.dumps(
            {
                "page_no": str(idx + 1),
                "page_type": "Bill Detail",
                "bill_items": [
                    {
                        "item_name": f"Mock Item {idx+1}",
                        "item_amount": 100.0 * (idx + 1),
                        "item_rate": 100.0,
                        "item_quantity": idx + 1,
                    }
                ],
                "fraud_flags": [],
            }
        )


# ── Main extractor ─────────────────────────────────────────────────────────

class BillExtractor:
    """
    End-to-end pipeline:
      PDF/image → page images → (OCR) → Gemini → structured JSON
    """

    def __init__(self):
        self.gemini = GeminiCaller()

    def extract(self, file_path: str) -> ExtractionResponse:
        # Fresh caller per extraction. BillExtractor is a module-level singleton
        # in app.main, so a caller reused across requests would accumulate token
        # counts and failure counters -- reporting "3/3 batches failed" for a
        # 2-page document because it was really counting three separate requests.
        self.gemini = GeminiCaller()
        try:
            return self._run(file_path)
        except Exception as e:
            logger.exception(f"Extraction failed: {e}")
            return ExtractionResponse(
                is_success=False,
                error=str(e),
            )

    def _run(self, file_path: str) -> ExtractionResponse:
        suffix = Path(file_path).suffix.lower()

        # ── Step 1: Get page images ────────────────────────────────────────
        if suffix == ".pdf":
            pil_images = _pdf_to_images(file_path)
            if not pil_images:
                # Try extracting text-based PDF and fake single-page image
                pil_images = self._text_pdf_fallback(file_path)
        else:
            if OCR_AVAILABLE:
                pil_images = [Image.open(file_path)]
            else:
                pil_images = []

        if not pil_images:
            return ExtractionResponse(
                is_success=False,
                error="Could not render any pages from the document.",
            )

        logger.info(f"Extracted {len(pil_images)} pages from {file_path}")

        # ── Step 2: Save images to temp JPEG + OCR ─────────────────────────
        page_b64s: List[str] = []
        page_ocr: List[str] = []

        # Pages are independent, and tesseract releases the GIL while the
        # binary runs, so a thread pool turns the slowest stage in the pipeline
        # from O(n) into O(n / workers). Order is restored by index afterwards.
        def _prepare(idx_img):
            i, img = idx_img
            # Two stages, in this order and both needed:
            #   OSD  - fixes 90/180/270 rotations (a sideways scan)
            #   deskew - fixes the remaining few degrees of tilt from a photo
            img, _deg = _autorotate(img)
            img, _tilt = _deskew(img)
            ocr_text = _ocr_page(img) if (OCR_AVAILABLE or OCR_ENGINE == "paddle") else ""
            return i, img, ocr_text

        with tempfile.TemporaryDirectory() as tmpdir:
            t_ocr = time.perf_counter()
            with ThreadPoolExecutor(max_workers=max(1, OCR_WORKERS)) as pool:
                prepared = list(pool.map(_prepare, enumerate(pil_images)))
            prepared.sort(key=lambda r: r[0])
            logger.info(
                "OCR+rotation for %d pages in %.1fs (%d workers)",
                len(pil_images), time.perf_counter() - t_ocr, OCR_WORKERS,
            )

            for i, img, ocr_text in prepared:
                img_path = os.path.join(tmpdir, f"page_{i+1}.jpg")
                try:
                    # The ROTATED image is what gets sent to the model, not just
                    # what gets OCR'd -- a sideways page hurts the VLM too.
                    img.convert("RGB").save(img_path, "JPEG")
                except Exception as e:  # noqa: BLE001
                    logger.warning(f"could not encode page {i+1}: {e}")
                    open(img_path, "wb").write(b"")

                page_b64s.append(_encode_image(img_path))
                page_ocr.append(ocr_text)

        # ── Step 3: Batch pages → Gemini ──────────────────────────────────
        all_page_results: List[PageLineItems] = []
        all_fraud_flags: List[str] = []
        empty_diagnostics: List[str] = []

        for batch_start in range(0, len(page_b64s), BATCH_SIZE):
            batch_b64 = page_b64s[batch_start: batch_start + BATCH_SIZE]
            batch_ocr = page_ocr[batch_start: batch_start + BATCH_SIZE]
            batch_nos = list(range(batch_start + 1, batch_start + len(batch_b64) + 1))

            raw_responses = self.gemini.call(batch_b64, batch_ocr, batch_nos)

            for page_no, raw in zip(batch_nos, raw_responses):
                page_result, fraud_flags = _parse_page_result(raw, page_no)
                all_page_results.append(page_result)
                all_fraud_flags.extend(fraud_flags)

                # Capture WHY a page came back empty. Without this the caller
                # sees a valid-looking response with no items and no clue.
                if not page_result.bill_items:
                    try:
                        keys = list(json.loads(_clean_json(raw)).keys())
                        empty_diagnostics.append(f"page {page_no}: keys={keys}")
                    except Exception:  # noqa: BLE001
                        empty_diagnostics.append(
                            f"page {page_no}: unparseable, starts {raw[:80]!r}"
                        )

        # ── Step 3b: Did the model actually answer? ────────────────────────
        # Every failed batch yields "{}" per page, which parses into zero items
        # without raising. Reporting that as a successful extraction of an empty
        # bill is worse than failing: downstream systems would record a
        # legitimate zero-rupee claim. Surface it as an error instead.
        gem = self.gemini
        if gem.calls_attempted and gem.calls_failed == gem.calls_attempted:
            return ExtractionResponse(
                is_success=False,
                token_usage=gem.token_usage,
                error=(
                    f"Extraction failed: every Gemini request failed "
                    f"({gem.calls_failed}/{gem.calls_attempted} batches). "
                    f"Last error -- {gem.last_error}"
                ),
            )

        # The model answered and cost tokens, but nothing parsed into items.
        # Surface it: an empty bill that reports success is indistinguishable
        # from a genuinely zero-value bill, and downstream that is dangerous.
        parse_warning = None
        total_parsed = sum(len(p.bill_items) for p in all_page_results)
        if total_parsed == 0 and gem.token_usage.output_tokens > 0:
            parse_warning = (
                f"Model returned {gem.token_usage.output_tokens:,} output tokens "
                f"but no line items parsed from any of {len(all_page_results)} "
                f"page(s). Response shapes seen -- "
                f"{'; '.join(empty_diagnostics[:6])}. "
                f"Try BATCH_SIZE=1: dense pages batched together are the usual cause."
            )
            logger.error(parse_warning)

        partial_warning = None
        if gem.calls_failed:
            partial_warning = (
                f"{gem.calls_failed} of {gem.calls_attempted} page batches "
                f"failed; results are incomplete. Last error -- {gem.last_error}"
            )
            logger.warning(partial_warning)

        if all_fraud_flags:
            logger.warning(f"FRAUD FLAGS detected: {all_fraud_flags}")

        # ── Step 4: Deduplicate & compute totals ───────────────────────────
        final_pages = self._deduplicate(all_page_results)
        total_items = sum(len(p.bill_items) for p in final_pages)
        grand_total = sum(
            item.item_amount
            for p in final_pages
            if p.page_type != "Bill Summary"
            for item in p.bill_items
        )

        reconciliation = self._reconcile(all_page_results, grand_total)
        if reconciliation.matches is False:
            logger.warning(
                "RECONCILIATION FAILED: extracted %.2f vs printed %.2f (%.1f%%)",
                reconciliation.computed_total,
                reconciliation.printed_total or 0.0,
                reconciliation.pct_difference or 0.0,
            )

        # De-duplicate flags while preserving order; the same observation often
        # repeats across pages of one bill.
        seen, unique_flags = set(), []
        for f in all_fraud_flags:
            if f not in seen:
                seen.add(f)
                unique_flags.append(f)

        return ExtractionResponse(
            is_success=True,
            token_usage=self.gemini.token_usage,
            # Populated only on PARTIAL failure: some pages came back, some did
            # not. is_success stays true because there is usable data, but the
            # caller must know it is incomplete.
            error=partial_warning or parse_warning,
            data=ExtractionData(
                pagewise_line_items=final_pages,
                total_item_count=total_items,
                grand_total=round(grand_total, 2),
                reconciliation=reconciliation,
                fraud_flags=unique_flags,
            ),
        )

    def _reconcile(self, pages: List[PageLineItems], computed: float) -> Reconciliation:
        """
        Compare extracted line items against the total printed on the bill.

        Uses the MAXIMUM printed total across pages, not the sum. A multi-page
        bill repeats a running or final total on several pages, and a summary
        page restates the whole bill -- adding them would produce a target
        several times too large. The largest printed figure is the best
        available estimate of the document's own grand total.

        This is imperfect: an interim bill can print a total larger than the
        pages provided (several of the sample documents are excerpts of longer
        bills), which shows up as a large negative difference. That is still
        useful signal -- it tells you the input was partial.
        """
        printed_values = [
            p.printed_total for p in pages
            if p.printed_total is not None and p.printed_total > 0
        ]
        computed = round(computed, 2)

        if not printed_values:
            return Reconciliation(
                printed_total=None,
                computed_total=computed,
                matches=None,
                tolerance_pct=RECONCILE_TOLERANCE_PCT,
                note="No total printed on the document; nothing to verify against.",
            )

        printed = round(max(printed_values), 2)
        diff = round(computed - printed, 2)
        pct = round(abs(diff) / printed * 100, 2) if printed else None
        ok = pct is not None and pct <= RECONCILE_TOLERANCE_PCT

        if ok:
            note = "Extracted items match the printed total."
        elif diff < 0:
            note = (
                f"Extracted total is {abs(diff):,.2f} LOWER than printed - line "
                f"items were likely missed, or the document is a partial excerpt."
            )
        else:
            note = (
                f"Extracted total is {diff:,.2f} HIGHER than printed - likely "
                f"double-counting, e.g. a summary page counted alongside detail pages."
            )

        return Reconciliation(
            printed_total=printed,
            computed_total=computed,
            difference=diff,
            pct_difference=pct,
            matches=ok,
            tolerance_pct=RECONCILE_TOLERANCE_PCT,
            note=note,
        )

    def _deduplicate(self, pages: List[PageLineItems]) -> List[PageLineItems]:
        """
        If the same bill has both a Summary page and Detail pages,
        keep the Detail pages (which have individual items) and
        suppress the Summary page's items to avoid double-counting.
        
        Strategy: If any page has page_type != 'Bill Summary', suppress
        all 'Bill Summary' pages' items (set to empty list but keep the page).
        """
        has_detail = any(p.page_type != "Bill Summary" for p in pages)
        if not has_detail:
            return pages

        result = []
        for p in pages:
            if p.page_type == "Bill Summary":
                # Keep the page in output (for reference) but clear items.
                # printed_total is preserved -- the summary page usually carries
                # the authoritative grand total, which reconciliation needs.
                result.append(
                    PageLineItems(
                        page_no=p.page_no,
                        page_type=p.page_type,
                        bill_items=[],
                        printed_total=p.printed_total,
                        fraud_flags=p.fraud_flags,
                    )
                )
            else:
                result.append(p)
        return result

    def _text_pdf_fallback(self, pdf_path: str) -> list:
        """
        If pdf2image is not available, extract text via pypdf and create
        a simple PIL image with the text drawn on it (for downstream LLM call).
        """
        try:
            from pypdf import PdfReader
            from PIL import Image as PILImage, ImageDraw, ImageFont

            reader = PdfReader(pdf_path)
            images = []
            for page in reader.pages:
                text = page.extract_text() or ""
                # Create white image with text
                img = PILImage.new("RGB", (1200, 1600), color="white")
                draw = ImageDraw.Draw(img)
                # Draw text in chunks
                lines = text.split("\n")
                y = 20
                for line in lines[:80]:
                    draw.text((20, y), line[:120], fill="black")
                    y += 18
                    if y > 1550:
                        break
                images.append(img)
            return images
        except Exception as e:
            logger.error(f"Text PDF fallback failed: {e}")
            return []
