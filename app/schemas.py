"""
Pydantic schemas for the Medical Bill Extractor API.
"""

from typing import List, Optional, Any
from pydantic import BaseModel, Field


class BillItem(BaseModel):
    item_name: str = Field(..., description="Name or description of the line item")
    item_amount: float = Field(..., description="Total amount for this line item")
    item_rate: Optional[float] = Field(None, description="Unit rate / price per unit")
    item_quantity: Optional[float] = Field(None, description="Quantity of this item")


class PageLineItems(BaseModel):
    page_no: str = Field(..., description="Page number (1-indexed string)")
    page_type: str = Field(
        ...,
        description=(
            "Type of bill page: 'Bill Summary', 'Bill Detail', "
            "'Pharmacy Bill', 'Lab Bill', 'Other'"
        ),
    )
    bill_items: List[BillItem] = Field(
        default_factory=list, description="Extracted line items from this page"
    )
    printed_total: Optional[float] = Field(
        None,
        description=(
            "Grand total as PRINTED on this page, copied verbatim. Null if the "
            "page shows no total. Used to verify the extracted line items."
        ),
    )
    fraud_flags: List[str] = Field(
        default_factory=list,
        description="Suspicious observations reported by the model for this page",
    )


class Reconciliation(BaseModel):
    """
    Cross-check of extracted line items against the total printed on the bill.

    This is the cheapest quality signal available: the document states its own
    answer, so we can detect dropped rows or double-counting per request without
    any labelled data. `matches` false means the extraction is provably wrong
    somewhere, which is far more actionable than a confidence score.
    """

    printed_total: Optional[float] = Field(
        None, description="Total printed on the bill (max across pages)"
    )
    computed_total: float = Field(
        0.0, description="Sum of extracted line items, excluding Bill Summary pages"
    )
    difference: Optional[float] = Field(
        None, description="computed_total - printed_total"
    )
    pct_difference: Optional[float] = Field(
        None, description="Difference as a percentage of the printed total"
    )
    matches: Optional[bool] = Field(
        None,
        description=(
            "True if within tolerance, False if not, None if the bill printed "
            "no total to compare against"
        ),
    )
    tolerance_pct: float = Field(
        1.0, description="Percentage tolerance used for the match decision"
    )
    note: Optional[str] = Field(None, description="Human-readable interpretation")


class ExtractionData(BaseModel):
    pagewise_line_items: List[PageLineItems] = Field(
        default_factory=list,
        description="List of pages with their extracted line items",
    )
    total_item_count: int = Field(
        0, description="Total count of all line items across all pages"
    )
    grand_total: Optional[float] = Field(
        None, description="Grand total as shown on the bill (avoid double counting)"
    )
    sub_totals: Optional[dict] = Field(
        None, description="Category-wise subtotals if present on the bill"
    )
    reconciliation: Optional[Reconciliation] = Field(
        None,
        description="Check of extracted items against the total printed on the bill",
    )
    fraud_flags: List[str] = Field(
        default_factory=list,
        description="All fraud observations across pages, de-duplicated",
    )


class TokenUsage(BaseModel):
    total_tokens: int = 0
    input_tokens: int = 0
    output_tokens: int = 0


class ExtractionResponse(BaseModel):
    is_success: bool
    token_usage: TokenUsage = Field(default_factory=TokenUsage)
    data: Optional[ExtractionData] = None
    error: Optional[str] = None
