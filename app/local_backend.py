"""
Local-VLM extraction backend, shaped to match BillExtractor's interface.

Lets the running service swap between the hosted API and the fine-tuned adapter
without touching route code:

    EXTRACTOR_BACKEND=gemini   # default, unchanged behaviour
    EXTRACTOR_BACKEND=local    # Qwen2-VL-2B + LoRA, no network egress

The second mode is the point of the whole exercise: no per-page API cost, no
PHI leaving the host, and a path to the same weights running inside a mobile
SDK. It needs a GPU (or a lot of patience on CPU) and the vlm/ requirements.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from app.schemas import BillItem, ExtractionData, ExtractionResponse, PageLineItems, TokenUsage

logger = logging.getLogger(__name__)


class LocalBillExtractor:
    """Mirror of BillExtractor.extract() backed by the local adapter."""

    def __init__(self, adapter: str | None = None):
        from vlm.serve.local_vlm import LocalVLM

        self.vlm = LocalVLM(adapter=adapter or os.getenv("VLM_ADAPTER", "stage_c"))

    def extract(self, file_path: str) -> ExtractionResponse:
        try:
            return self._run(file_path)
        except Exception as e:  # noqa: BLE001
            logger.exception("Local extraction failed")
            return ExtractionResponse(is_success=False, error=str(e))

    def _run(self, file_path: str) -> ExtractionResponse:
        import tempfile

        from vlm.data.render import render_pdf

        suffix = Path(file_path).suffix.lower()
        if suffix == ".pdf":
            with tempfile.TemporaryDirectory() as tmp:
                page_paths = render_pdf(Path(file_path), Path(tmp), dpi=200)
                pages = self._pages(page_paths)
        else:
            pages = self._pages([Path(file_path)])

        if not pages:
            return ExtractionResponse(
                is_success=False, error="Could not render any pages from the document."
            )

        total_items = sum(len(p.bill_items) for p in pages)
        grand_total = sum(
            i.item_amount for p in pages if p.page_type != "Bill Summary" for i in p.bill_items
        )

        return ExtractionResponse(
            is_success=True,
            # Local inference consumes no billable tokens. Reporting zeros here
            # is the accurate answer, not a missing value.
            token_usage=TokenUsage(),
            data=ExtractionData(
                pagewise_line_items=pages,
                total_item_count=total_items,
                grand_total=round(grand_total, 2),
            ),
        )

    def _pages(self, paths: list[Path]) -> list[PageLineItems]:
        out = []
        for i, p in enumerate(paths, start=1):
            data, _ = self.vlm.extract(str(p))
            out.append(
                PageLineItems(
                    page_no=str(i),
                    page_type=data["page_type"],
                    bill_items=[BillItem(**{k: v for k, v in it.items()}) for it in data["bill_items"]],
                )
            )
        return out


def build_extractor():
    """Factory used by app.main. Falls back to Gemini with a loud log line."""
    backend = os.getenv("EXTRACTOR_BACKEND", "gemini").lower()
    if backend == "local":
        try:
            ex = LocalBillExtractor()
            logger.info("extractor backend: local VLM (adapter=%s)", ex.vlm.adapter_path)
            return ex
        except Exception as e:  # noqa: BLE001
            logger.error("local backend unavailable (%s); falling back to Gemini", e)

    from app.extractor import BillExtractor

    logger.info("extractor backend: gemini")
    return BillExtractor()
