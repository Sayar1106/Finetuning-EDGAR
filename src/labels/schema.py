"""The *extraction target* schema -- the JSON the fine-tuned model is trained to emit.

Deliberately separate from src/data/schema.py, which describes raw ingested material. This module
is the product: given filing text, produce this shape. The eval harness (src/eval/) scores model
output against it field by field.
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field

from src.data.schema import FinancialFacts

# Fixed taxonomy. A closed set makes risk-factor categorization scoreable as multi-label F1 rather
# than as free text -- open-vocabulary categories would collapse the metric into fuzzy matching.
RiskCategory = Literal[
    "market",
    "operational",
    "regulatory",
    "legal",
    "cyber",
    "financial",
    "supply_chain",
    "talent",
    "climate",
    "competitive",
    "intellectual_property",
    "macroeconomic",
]

RISK_CATEGORIES: tuple[str, ...] = (
    "market",
    "operational",
    "regulatory",
    "legal",
    "cyber",
    "financial",
    "supply_chain",
    "talent",
    "climate",
    "competitive",
    "intellectual_property",
    "macroeconomic",
)


class RiskFactor(BaseModel):
    """One risk disclosed in Item 1A."""

    title: str = Field(description="Short label for the risk, under 12 words.")
    category: RiskCategory = Field(description="Single best-fitting category from the taxonomy.")
    summary: str = Field(description="One or two sentences on what the risk is and why it matters.")


class RiskFactorList(BaseModel):
    """Teacher-model response envelope. The API's structured-output mode requires an object at the
    top level, so the list is wrapped rather than returned bare."""

    risk_factors: list[RiskFactor]


class ExtractionTarget(BaseModel):
    """The full JSON the student model produces for one filing."""

    company: str
    ticker: str
    fiscal_year: Optional[int] = None
    form_type: str
    financials: FinancialFacts
    risk_factors: list[RiskFactor] = []
