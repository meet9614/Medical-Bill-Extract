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


class TokenUsage(BaseModel):
    total_tokens: int = 0
    input_tokens: int = 0
    output_tokens: int = 0


class ExtractionResponse(BaseModel):
    is_success: bool
    token_usage: TokenUsage = Field(default_factory=TokenUsage)
    data: Optional[ExtractionData] = None
    error: Optional[str] = None
