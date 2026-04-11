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
from pathlib import Path
from typing import List, Optional, Tuple

from dotenv import load_dotenv

load_dotenv()

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


def _ocr_page(img) -> str:
    """Run Tesseract on a PIL image and return raw text."""
    if not OCR_AVAILABLE:
        return ""
    try:
        enhanced = _enhance_image(img.copy())
        text = pytesseract.image_to_string(enhanced, config="--psm 6")
        return text.strip()
    except Exception as e:
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
            parts.append(
                {"text": f"\n\n--- PAGE {pg_no} ---\nOCR HINT:\n{ocr_text[:1500] if ocr_text else '(no OCR)'}\n\nReturn JSON for this page only:\n"}
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

        with tempfile.TemporaryDirectory() as tmpdir:
            for i, img in enumerate(pil_images):
                img_path = os.path.join(tmpdir, f"page_{i+1}.jpg")
                # Ensure RGB for JPEG
                if OCR_AVAILABLE:
                    ocr_text = _ocr_page(img)
                    rgb = img.convert("RGB")
                    rgb.save(img_path, "JPEG")
                else:
                    ocr_text = ""
                    # Minimal save without OCR
                    try:
                        img.save(img_path, "JPEG")
                    except Exception:
                        # Create blank placeholder
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
