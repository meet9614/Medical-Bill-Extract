"""Pydantic schemas for the medical bill extractor.
"""
from pydantic import BaseModel
from typing import Dict, Any, Optional, List


class ExtractionRequest(BaseModel):
    """Request schema for text-based extraction.

    You can send either `text` (raw extracted text from a PDF) or a `filename` as context.
    """
    text: Optional[str] = None
    filename: Optional[str] = None


class ExtractionResult(BaseModel):
    """Response schema for extraction results.

    fields: a mapping of extracted field name -> value
    confidence: optional overall confidence score (0.0 - 1.0)
    warnings: optional list of warnings or notes
    """
    fields: Dict[str, Any]
    confidence: Optional[float] = None
    warnings: Optional[List[str]] = []


class HealthRecord(BaseModel):
    """Example schema representing a simplified medical bill record."""
    patient_name: Optional[str] = None
    invoice_number: Optional[str] = None
    provider: Optional[str] = None
    date: Optional[str] = None
    total_amount: Optional[str] = None


# End of schemas.py