"""Data shapes produced by the ingestion pipeline (src/data/) and consumed by label generation
(src/labels/). Kept intentionally separate from the *extraction target* schema the fine-tuned
model will produce -- this is raw ingested material, not training labels yet.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel


class FinancialFacts(BaseModel):
    """Numeric ground truth pulled from XBRL. `None` means the concept was not reported by this
    filer for this period -- treat as missing, not as zero."""

    revenue: Optional[float] = None
    net_income: Optional[float] = None
    total_assets: Optional[float] = None
    eps_diluted: Optional[float] = None
    shares_outstanding_diluted: Optional[float] = None
    unit: str = "USD"


class FilingSections(BaseModel):
    """Plain-text sections pulled from a single 10-K, plus the XBRL facts for the same period."""

    ticker: str
    cik: int
    company_name: str
    accession_no: str
    form_type: str
    fiscal_year: Optional[int] = None
    period_of_report: Optional[str] = None
    filing_date: Optional[str] = None

    business: Optional[str] = None
    risk_factors: Optional[str] = None
    management_discussion: Optional[str] = None

    # Item 8. Most numeric ground truth appears only here -- not in the narrative sections.
    income_statement: Optional[str] = None
    balance_sheet: Optional[str] = None

    financials: FinancialFacts = FinancialFacts()

    parse_warnings: list[str] = []
