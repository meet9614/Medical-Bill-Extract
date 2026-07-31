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

print("📁 ENV PATH:", ENV_PATH)

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
    TokenUsage,
)

# ── Config ─────────────────────────────────────────────────────────────────
GOOGLE_API_KEY   = os.getenv("GOOGLE_API_KEY", "")
print("🔑 API KEY:", GOOGLE_API_KEY[:5] if GOOGLE_API_KEY else "❌ NOT LOADED")
GEMINI_MODEL     = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
FALLBACK_MODELS  = [
    m.strip()
    for m in os.getenv(
        "GEMINI_MODEL_FALLBACKS", "gemini-2.0-flash,gemini-1.5-flash"
    ).split(",")
    if m.strip()
]
MAX_RETRIES      = int(os.getenv("MAX_RETRIES", "5"))
BATCH_SIZE       = int(os.getenv("BATCH_SIZE", "3"))
PDF_DPI          = int(os.getenv("PDF_DPI", "200"))
USE_MOCK_MODE    = os.getenv("USE_MOCK_MODE", "false").lower() == "true"

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
# 35 therefore only catches pages where OCR has collapsed completely. Whether
# dropping hints on handwriting actually helps is an open question -- settle it
# with `python -m vlm.eval.ocr_ab --mode ab`, not with a guessed threshold.
OCR_MIN_CONFIDENCE = float(os.getenv("OCR_MIN_CONFIDENCE", "35"))

# Master switch for the A/B: does the OCR hint earn its latency at all?
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
8. Fraud indicators to flag: mismatched fonts, white-out / whitener over text,
   amounts that do not match rate × quantity, duplicate serial numbers.
9. Return ONLY valid JSON – no markdown fences, no prose.
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

    page_line = PageLineItems(
        page_no=str(data.get("page_no", page_no)),
        page_type=str(data.get("page_type", "Bill Detail")),
        bill_items=items,
    )
    fraud_flags = data.get("fraud_flags", [])
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
        if USE_MOCK_MODE:
            return [self._mock_response(i) for i in range(len(page_images))]

        if not GEMINI_AVAILABLE or not GOOGLE_API_KEY:
            logger.error("Gemini not configured – set GOOGLE_API_KEY")
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
        for model_name in self.model_queue:
            for attempt in range(MAX_RETRIES):
                try:
                    model = genai.GenerativeModel(model_name)
                    response = model.generate_content(
                        parts,
                        generation_config=genai.types.GenerationConfig(
                            temperature=0.1,
                            max_output_tokens=8192,
                        ),
                    )
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
                    wait = 2 ** attempt
                    logger.warning(
                        f"Gemini {model_name} attempt {attempt+1} failed: {e}. Retrying in {wait}s"
                    )
                    time.sleep(wait)

        logger.error(f"All Gemini models failed. Last error: {last_error}")
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
                return [json.dumps(item) for item in arr]
            if isinstance(arr, dict):
                return [json.dumps(arr)]
        except json.JSONDecodeError:
            pass

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
            img, _deg = _autorotate(img)
            ocr_text = _ocr_page(img) if OCR_AVAILABLE else ""
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

        for batch_start in range(0, len(page_b64s), BATCH_SIZE):
            batch_b64 = page_b64s[batch_start: batch_start + BATCH_SIZE]
            batch_ocr = page_ocr[batch_start: batch_start + BATCH_SIZE]
            batch_nos = list(range(batch_start + 1, batch_start + len(batch_b64) + 1))

            raw_responses = self.gemini.call(batch_b64, batch_ocr, batch_nos)

            for page_no, raw in zip(batch_nos, raw_responses):
                page_result, fraud_flags = _parse_page_result(raw, page_no)
                all_page_results.append(page_result)
                all_fraud_flags.extend(fraud_flags)

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

        return ExtractionResponse(
            is_success=True,
            token_usage=self.gemini.token_usage,
            data=ExtractionData(
                pagewise_line_items=final_pages,
                total_item_count=total_items,
                grand_total=round(grand_total, 2),
            ),
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
                # Keep the page in output (for reference) but clear items
                result.append(
                    PageLineItems(
                        page_no=p.page_no,
                        page_type=p.page_type,
                        bill_items=[],
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
