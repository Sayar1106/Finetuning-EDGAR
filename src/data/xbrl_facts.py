"""Numeric ground truth for the extraction schema, sourced from XBRL rather than model/teacher
generation. This is the "clever labeling trick" from the project plan: financial figures are
exact and free because SEC filers are required to tag them.

"Exact" only holds if the *right* fact is selected. A concept query returns every fact filed under
that tag -- across reporting periods, across segment dimensions, and including text-valued facts
that carry no number at all. Taking the first row, as this module originally did, produced silently
wrong labels on 15 of 182 filings:

  - **Dimensional facts.** EA's first `EarningsPerShareDiluted` row is a segment breakdown labeled
    "Performance Obligation Period" (-0.22); the headline 3.51 is the second row.
  - **Text-valued facts.** Wells Fargo's first three EPS rows are "Extensible Enumeration" entries
    whose numeric_value is NaN. `float(nan)` succeeds, so NaN propagated into the labels as though
    it were a figure -- and NaN silently compares unequal to everything, so it never scores wrong.
  - **The convenience dict.** `Financials.get_financial_metrics()["revenue"]` disagrees with the
    filed `us-gaap:Revenues` tag for every bank and insurer in the corpus (Wells Fargo: 16.1B vs
    the reported 83.7B). The raw tag matches the printed income statement; the dict does not.
  - **Substring concept matching.** `by_concept()` matches by prefix unless `exact=True`, so a
    query for `NetIncomeLoss` also returns `NetIncomeLossAttributableToNoncontrollingInterest` --
    AbbVie's 7M noncontrolling-interest line sorts ahead of its 4,226M net income.
  - **Interim periods.** A 10-K tags Q4 alongside the full year, and both share the fiscal
    year-end as `period_end`. AbbVie files 1,816M (Q4) and 4,226M (FY) against the same date, so
    matching on `period_end` alone is not enough -- the annual figure is the longest duration
    ending on that date.

So facts are now selected explicitly: exact concept, undimensioned, numeric, ending on the
filing's own reporting date, and spanning the longest period that does. The grounding check in
src/labels/excerpt.py is what surfaced these -- a label that never appears in the filing text is
the signature of a mis-selected fact.
"""

from __future__ import annotations

import logging
import math

import pandas as pd

from src.data.schema import FinancialFacts

logger = logging.getLogger(__name__)

# Tried in order; the first concept that yields a usable fact wins. `Revenues` leads because it is
# the total-revenue tag filers use on the face of the income statement, including the financials
# whose revenue the convenience dict gets wrong.
_CONCEPTS: dict[str, tuple[str, ...]] = {
    "revenue": (
        "us-gaap:Revenues",
        "us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax",
        "us-gaap:RevenuesNetOfInterestExpense",
        "us-gaap:SalesRevenueNet",
    ),
    "net_income": (
        "us-gaap:NetIncomeLoss",
        "us-gaap:ProfitLoss",
    ),
    "total_assets": ("us-gaap:Assets",),
    "eps_diluted": (
        "us-gaap:EarningsPerShareDiluted",
        "us-gaap:IncomeLossFromContinuingOperationsPerDilutedShare",
    ),
}

# Facts on the face of a primary statement are the headline figures; note disclosures repeat the
# same concepts with narrower meanings.
_PREFERRED_STATEMENTS = ("IncomeStatement", "BalanceSheet")


def _is_missing(value) -> bool:
    """NaN counts as missing. It is not None, and float(nan) succeeds, so an explicit check is the
    only thing standing between a text-valued XBRL fact and a NaN label."""
    if value is None:
        return True
    try:
        return math.isnan(float(value))
    except (TypeError, ValueError):
        return True


def _period_end(row) -> str:
    """The date a fact is reported as of.

    Duration facts (income statement) carry `period_start`/`period_end`; instant facts (balance
    sheet) carry `period_instant` instead, and the two kinds do not even share dataframe columns.
    Checking only `period_end` silently rejected every balance-sheet fact in the corpus, which sent
    `total_assets` to the convenience-dict fallback for all 182 filings.
    """
    for column in ("period_end", "period_instant"):
        value = row.get(column)
        if value is not None and not _is_blank(value):
            return str(value)
    return ""


def _is_blank(value) -> bool:
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):  # pragma: no cover - non-scalar values
        return False


def _duration_days(row) -> int:
    """Length of the row's reporting period. A 10-K's annual figure is the longest duration ending
    on the fiscal year-end; Q4 shares that end date with a ~90-day span. Instants score 0 -- they
    never compete with durations for the same field."""
    start, end = row.get("period_start"), row.get("period_end")
    if start is None or end is None or _is_blank(start) or _is_blank(end):
        return 0
    try:
        return (pd.Timestamp(end) - pd.Timestamp(start)).days
    except (TypeError, ValueError):
        return 0


def _query(xb, concept: str):
    """`exact=True` matters: the default matches by prefix, so `NetIncomeLoss` would also return
    `NetIncomeLossAttributableToNoncontrollingInterest`."""
    try:
        return xb.query().by_concept(concept, exact=True).to_dataframe()
    except TypeError:  # pragma: no cover - edgartools versions without the `exact` parameter
        return xb.query().by_concept(concept).to_dataframe()


def _select_fact(xb, concept: str, period_end: str | None) -> float | None:
    """The undimensioned, numeric, full-period fact for `concept` on this filing's own period."""
    try:
        df = _query(xb, concept)
    except Exception as e:  # pragma: no cover - defensive against library API drift
        logger.debug("XBRL query failed for concept %s: %s", concept, e)
        return None
    if df is None or df.empty:
        return None

    rows = [r for _, r in df.iterrows() if not _is_missing(r.get("numeric_value"))]
    # Segment/axis breakdowns share the concept tag with the consolidated figure.
    rows = [r for r in rows if not bool(r.get("is_dimensioned", False))]
    if not rows:
        return None

    if period_end:
        matching = [r for r in rows if _period_end(r) == str(period_end)]
        # A filing with no fact for its own period is a data problem, not a reason to grab a prior
        # year's number and label it as this year's.
        if not matching:
            return None
        rows = matching

    on_statement = [r for r in rows if r.get("statement_type") in _PREFERRED_STATEMENTS]
    rows = on_statement or rows

    # Longest period wins: the annual figure over the fourth quarter that ends on the same day.
    rows.sort(key=_duration_days, reverse=True)

    try:
        return float(rows[0].get("numeric_value"))
    except (TypeError, ValueError):  # pragma: no cover - guarded by _is_missing above
        return None


def extract_financial_facts(filing) -> tuple[FinancialFacts, list[str]]:
    """`filing` is an edgar.Filing (10-K). Returns best-effort ground truth; missing values are
    left as None rather than guessed."""

    warnings: list[str] = []
    period_end = getattr(filing, "period_of_report", None)
    period_end = str(period_end) if period_end else None

    values: dict[str, float | None] = dict.fromkeys(_CONCEPTS, None)
    try:
        xb = filing.xbrl()
        for field, concepts in _CONCEPTS.items():
            for concept in concepts:
                value = _select_fact(xb, concept, period_end)
                if value is not None:
                    values[field] = value
                    break
    except Exception as e:
        logger.warning("XBRL unavailable for %s: %s", filing.accession_no, e)
        warnings.append(f"xbrl_unavailable: {e}")

    # The convenience dict is the fallback, not the primary source -- it is wrong for financials
    # (see module docstring), but it is better than nothing when a concept is absent entirely.
    metrics: dict = {}
    if any(v is None for v in values.values()):
        try:
            metrics = filing.obj().financials.get_financial_metrics() or {}
        except Exception as e:
            logger.debug("Financial metrics unavailable for %s: %s", filing.accession_no, e)

    for field in ("revenue", "net_income", "total_assets"):
        if values[field] is None and not _is_missing(metrics.get(field)):
            values[field] = float(metrics[field])
            warnings.append(f"{field}_from_metrics_fallback")

    shares_diluted = metrics.get("shares_outstanding_diluted")
    if _is_missing(shares_diluted):
        shares_diluted = None

    if values["eps_diluted"] is None and values["net_income"] is not None and shares_diluted:
        values["eps_diluted"] = round(values["net_income"] / shares_diluted, 2)
        warnings.append("eps_diluted_derived_from_net_income_and_shares")

    facts = FinancialFacts(
        revenue=values["revenue"],
        net_income=values["net_income"],
        total_assets=values["total_assets"],
        eps_diluted=values["eps_diluted"],
        shares_outstanding_diluted=shares_diluted,
    )
    return facts, warnings
