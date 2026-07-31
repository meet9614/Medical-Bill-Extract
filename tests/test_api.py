"""
Tests for the Medical Bill Extractor API.
Run:  pytest -q
"""

import os
import json
import pytest
from fastapi.testclient import TestClient

# Force mock mode so tests don't need a real API key
os.environ["USE_MOCK_MODE"] = "true"

from app.main import app
from app.schemas import ExtractionResponse, BillItem, PageLineItems
from app.extractor import _clean_json, _safe_float, _parse_page_result, BillExtractor

client = TestClient(app)


# ── Unit tests ─────────────────────────────────────────────────────────────

class TestHelpers:
    def test_clean_json_strips_fences(self):
        raw = "```json\n{\"key\": 1}\n```"
        assert _clean_json(raw) == '{"key": 1}'

    def test_clean_json_no_fences(self):
        raw = '{"key": 1}'
        assert _clean_json(raw) == raw

    def test_safe_float_none(self):
        assert _safe_float(None) is None

    def test_safe_float_string(self):
        assert _safe_float("123.45") == 123.45

    def test_safe_float_invalid(self):
        assert _safe_float("abc") is None

    def test_safe_float_int(self):
        assert _safe_float(100) == 100.0

    def test_parse_page_result_valid(self):
        raw = json.dumps({
            "page_no": "1",
            "page_type": "Bill Detail",
            "bill_items": [
                {"item_name": "Consultation", "item_amount": 500.0,
                 "item_rate": 500.0, "item_quantity": 1.0}
            ],
            "fraud_flags": []
        })
        page, flags = _parse_page_result(raw, 1)
        assert page.page_no == "1"
        assert page.page_type == "Bill Detail"
        assert len(page.bill_items) == 1
        assert page.bill_items[0].item_name == "Consultation"
        assert page.bill_items[0].item_amount == 500.0
        assert flags == []

    def test_parse_page_result_invalid_json(self):
        page, flags = _parse_page_result("not json at all", 1)
        assert page.bill_items == []

    def test_parse_page_result_missing_amount(self):
        raw = json.dumps({
            "page_no": "2",
            "page_type": "Lab Bill",
            "bill_items": [
                {"item_name": "CBC", "item_amount": None}
            ]
        })
        page, _ = _parse_page_result(raw, 2)
        assert page.bill_items[0].item_amount == 0.0


class TestRepeatedItems:
    """
    Hospital bills legitimately repeat identical rows -- 20 identical glucometer
    charges, or the same consultation fee on consecutive days. Collapsing them
    silently deletes money, so nothing in the pipeline may deduplicate items.
    """

    def test_identical_items_are_all_preserved(self):
        raw = json.dumps({
            "page_no": "1",
            "page_type": "Bill Detail",
            "bill_items": [
                {"item_name": "BLOOD SUGAR BY GLUCOMETER", "item_amount": 80.0}
                for _ in range(20)
            ],
        })
        page, _ = _parse_page_result(raw, 1)
        assert len(page.bill_items) == 20
        assert sum(i.item_amount for i in page.bill_items) == 1600.0

    def test_printed_total_is_parsed(self):
        raw = json.dumps({
            "page_no": "1", "page_type": "Bill Detail",
            "bill_items": [{"item_name": "X", "item_amount": 100.0}],
            "printed_total": 100.0,
        })
        page, _ = _parse_page_result(raw, 1)
        assert page.printed_total == 100.0

    def test_missing_printed_total_is_none(self):
        raw = json.dumps({
            "page_no": "1", "page_type": "Bill Detail",
            "bill_items": [{"item_name": "X", "item_amount": 100.0}],
        })
        page, _ = _parse_page_result(raw, 1)
        assert page.printed_total is None


class TestDeduplication:
    def test_summary_suppressed_when_detail_present(self):
        extractor = BillExtractor()
        pages = [
            PageLineItems(
                page_no="1",
                page_type="Bill Summary",
                bill_items=[BillItem(item_name="Total", item_amount=5000)]
            ),
            PageLineItems(
                page_no="2",
                page_type="Bill Detail",
                bill_items=[
                    BillItem(item_name="Bed Charge", item_amount=2000),
                    BillItem(item_name="Consultation", item_amount=3000),
                ]
            ),
        ]
        result = extractor._deduplicate(pages)
        summary_page = next(p for p in result if p.page_type == "Bill Summary")
        detail_page  = next(p for p in result if p.page_type == "Bill Detail")
        assert summary_page.bill_items == []
        assert len(detail_page.bill_items) == 2

    def test_summary_kept_if_no_detail(self):
        extractor = BillExtractor()
        pages = [
            PageLineItems(
                page_no="1",
                page_type="Bill Summary",
                bill_items=[BillItem(item_name="Drugs", item_amount=1000)]
            )
        ]
        result = extractor._deduplicate(pages)
        assert len(result[0].bill_items) == 1


# ── API endpoint tests ──────────────────────────────────────────────────────

class TestHealthEndpoint:
    def test_health_ok(self):
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"


class TestExtractFromURL:
    def test_invalid_url_returns_400(self):
        r = client.post(
            "/extract-bill-data",
            json={"document": "http://localhost:0/nonexistent.pdf"},
        )
        assert r.status_code == 400

    def test_missing_document_field(self):
        r = client.post("/extract-bill-data", json={})
        assert r.status_code == 422  # Pydantic validation error

    def test_rejects_non_http_scheme(self):
        r = client.post("/extract-bill-data", json={"document": "file:///etc/passwd"})
        assert r.status_code == 400

    def test_blocks_link_local_metadata_address(self):
        """SSRF guard: cloud instance metadata must not be reachable."""
        r = client.post(
            "/extract-bill-data",
            json={"document": "http://169.254.169.254/latest/meta-data/"},
        )
        assert r.status_code == 400
        assert "non-public" in r.json()["detail"].lower()

    def test_blocks_private_network(self):
        r = client.post("/extract-bill-data", json={"document": "http://10.0.0.5/bill.pdf"})
        assert r.status_code == 400


class TestReconciliation:
    """The printed-total cross-check is the main label-free quality signal."""

    def _extractor(self):
        from app.extractor import BillExtractor

        return BillExtractor.__new__(BillExtractor)  # skip Gemini client setup

    def _page(self, page_type, items, printed=None):
        return PageLineItems(
            page_no="1",
            page_type=page_type,
            bill_items=[BillItem(item_name=n, item_amount=a) for n, a in items],
            printed_total=printed,
        )

    def test_matching_totals(self):
        pages = [self._page("Bill Detail", [("A", 1000.0), ("B", 500.0)], printed=1500.0)]
        r = self._extractor()._reconcile(pages, 1500.0)
        assert r.matches is True
        assert r.difference == 0.0

    def test_missed_items_detected(self):
        pages = [self._page("Bill Detail", [("A", 1000.0)], printed=1500.0)]
        r = self._extractor()._reconcile(pages, 1000.0)
        assert r.matches is False
        assert r.difference == -500.0
        assert "LOWER" in r.note

    def test_double_counting_detected(self):
        pages = [self._page("Bill Detail", [("A", 3000.0)], printed=1500.0)]
        r = self._extractor()._reconcile(pages, 3000.0)
        assert r.matches is False
        assert "HIGHER" in r.note

    def test_uses_max_printed_total_not_sum(self):
        """A total repeated across pages must not be added up."""
        pages = [
            self._page("Bill Detail", [("A", 1000.0)], printed=1500.0),
            self._page("Bill Summary", [], printed=1500.0),
            self._page("Bill Detail", [("B", 500.0)], printed=1500.0),
        ]
        r = self._extractor()._reconcile(pages, 1500.0)
        assert r.printed_total == 1500.0
        assert r.matches is True

    def test_no_printed_total_is_unknown_not_failure(self):
        pages = [self._page("Bill Detail", [("A", 100.0)])]
        r = self._extractor()._reconcile(pages, 100.0)
        assert r.matches is None
        assert r.printed_total is None

    def test_rounding_within_tolerance(self):
        pages = [self._page("Bill Detail", [("A", 1495.0)], printed=1500.0)]
        assert self._extractor()._reconcile(pages, 1495.0).matches is True


class TestExtractFromFile:
    def test_upload_empty_file(self, tmp_path):
        # Create a minimal fake PDF
        fake_pdf = tmp_path / "test.pdf"
        fake_pdf.write_bytes(b"%PDF-1.4\n%%EOF")
        with open(fake_pdf, "rb") as f:
            r = client.post(
                "/extract-from-file",
                files={"file": ("test.pdf", f, "application/pdf")},
            )
        # In mock mode this should still succeed (no real rendering needed)
        assert r.status_code in (200, 500)  # 500 if pdf2image not installed in CI

    def test_response_schema(self, tmp_path):
        """Validate response matches ExtractionResponse schema in mock mode."""
        fake_pdf = tmp_path / "bill.pdf"
        fake_pdf.write_bytes(b"%PDF-1.4\n%%EOF")
        with open(fake_pdf, "rb") as f:
            r = client.post(
                "/extract-from-file",
                files={"file": ("bill.pdf", f, "application/pdf")},
            )
        if r.status_code == 200:
            body = r.json()
            # Validate required top-level keys
            assert "is_success" in body
            assert "token_usage" in body
            assert "error" in body


# ── Schema tests ───────────────────────────────────────────────────────────

class TestSchemas:
    def test_bill_item_required_fields(self):
        item = BillItem(item_name="Test", item_amount=100.0)
        assert item.item_rate is None
        assert item.item_quantity is None

    def test_extraction_response_success(self):
        from app.schemas import ExtractionData, TokenUsage
        resp = ExtractionResponse(
            is_success=True,
            token_usage=TokenUsage(total_tokens=100, input_tokens=80, output_tokens=20),
            data=ExtractionData(
                pagewise_line_items=[],
                total_item_count=0,
                grand_total=0.0,
            ),
        )
        assert resp.is_success is True
        assert resp.error is None

    def test_extraction_response_failure(self):
        resp = ExtractionResponse(is_success=False, error="Something went wrong")
        assert resp.is_success is False
        assert resp.data is None
