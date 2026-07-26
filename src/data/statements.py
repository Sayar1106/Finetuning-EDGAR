"""Render Item 8 financial statements to compact text.

Added after the Day 3-4 grounding check showed that `net_income`, `total_assets`, and `eps_diluted`
appear nowhere in Items 1/1A/7 -- they live only in the financial statements. Without this module
those labels are unlearnable: the model would be trained to produce numbers its input never
contains, which teaches confident hallucination rather than extraction.

Rendering the statements also sharpens the task. Filers use their own line-item wording -- "Net
sales" vs "Total revenues" vs "Net revenue" -- so mapping a company's labels onto the canonical
schema is real normalization work, and XBRL supplies the answer key for free.
"""

from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)

_PERIOD_COL = re.compile(r"^\d{4}-\d{2}-\d{2}")
MAX_ROWS = 60


def _period_columns(df) -> list[str]:
    return [c for c in df.columns if _PERIOD_COL.match(str(c))]


def render_statement(statement, title: str) -> str | None:
    """Render one edgartools `Statement` as `label: value` lines for the most recent period.

    Values are written in full dollars with thousands separators, matching the XBRL facts used as
    ground truth, so a correct extraction is a lookup rather than a unit conversion.
    """
    if statement is None:
        return None
    try:
        df = statement.to_dataframe()
    except Exception as e:  # pragma: no cover - defensive against library API drift
        logger.debug("Could not render %s: %s", title, e)
        return None

    if df is None or df.empty:
        return None

    periods = _period_columns(df)
    if not periods:
        return None
    current = periods[0]

    lines = [f"## {title} (period ending {current})"]
    for _, row in df.iterrows():
        if row.get("abstract", False) or row.get("dimension", False):
            continue  # section headers and segment breakdowns, not line items
        label = str(row.get("label") or "").strip()
        value = row.get(current)
        if not label or value is None:
            continue
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        if numeric != numeric:  # NaN
            continue
        lines.append(f"{label}: {numeric:,.2f}".replace(".00", ""))
        if len(lines) > MAX_ROWS:
            break

    return "\n".join(lines) if len(lines) > 1 else None


def extract_statements(tenk) -> tuple[str | None, str | None, list[str]]:
    """Returns (income_statement_text, balance_sheet_text, warnings)."""
    warnings: list[str] = []

    income = None
    balance = None
    try:
        income = render_statement(tenk.income_statement, "Consolidated Statements of Operations")
    except Exception as e:
        warnings.append(f"income_statement_failed: {e}")
    try:
        balance = render_statement(tenk.balance_sheet, "Consolidated Balance Sheets")
    except Exception as e:
        warnings.append(f"balance_sheet_failed: {e}")

    if income is None:
        warnings.append("missing_income_statement")
    if balance is None:
        warnings.append("missing_balance_sheet")
    return income, balance, warnings
