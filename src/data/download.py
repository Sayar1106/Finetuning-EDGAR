"""Fetch and parse 10-K filings via edgartools.

edgartools already exposes clean, pre-split plain-text sections (Item 1 / 1A / 7) through
`Filing.obj()` -> `TenK`, so this module is a thin, defensive wrapper around that rather than a
from-scratch HTML parser: retry on transient errors, log and skip on permanent ones, and attach
XBRL-derived financial facts (src/data/xbrl_facts.py) to each parsed filing.
"""

from __future__ import annotations

import logging
import time

from src.data.schema import FilingSections
from src.data.statements import extract_statements
from src.data.xbrl_facts import extract_financial_facts

logger = logging.getLogger(__name__)


def _permanent_error_types() -> tuple[type[BaseException], ...]:
    """edgartools raises CompanyNotFoundError for unresolvable tickers; tolerate its absence on
    versions that don't export it rather than hard-failing at import time."""
    try:
        from edgar.entity.core import CompanyNotFoundError

        return (CompanyNotFoundError,)
    except ImportError:  # pragma: no cover - depends on edgartools version
        return ()


_PERMANENT_ERRORS = _permanent_error_types()


class FilingFetchError(Exception):
    """Raised when a filing can't be fetched/parsed after retries. Caller should log and skip."""


class PermanentFetchError(FilingFetchError):
    """The ticker will never resolve -- delisted, acquired, or simply wrong. Retrying cannot help."""


def _with_retry(fn, *, retries: int = 3, base_delay: float = 1.5, desc: str = "request"):
    last_exc = None
    for attempt in range(retries):
        try:
            return fn()
        except _PERMANENT_ERRORS as e:
            # A ticker that isn't in EDGAR's map stays missing. Three backoff rounds per ticker
            # cost ~10s each and can only produce the same failure.
            raise PermanentFetchError(f"{desc}: {e}") from e
        except Exception as e:  # SEC rate limits / transient network errors
            last_exc = e
            delay = base_delay * (2**attempt)
            logger.warning("%s failed (attempt %d/%d): %s -- retrying in %.1fs", desc, attempt + 1, retries, e, delay)
            time.sleep(delay)
    raise FilingFetchError(f"{desc} failed after {retries} attempts") from last_exc


def get_latest_10ks(ticker: str, n: int = 1):
    """Returns up to `n` most recent 10-K Filing objects for `ticker`, newest first."""
    import edgar

    def _fetch():
        company = edgar.Company(ticker)
        # `amendments=False` excludes 10-K/A. An amendment restates only the items it revises --
        # AMD's most recent one carries just Items 7 and 15 -- so treating it as the annual report
        # yields a filing with no risk factors and no statements, which looks like a parse failure
        # but is the document faithfully rendered.
        filings = company.get_filings(form="10-K", amendments=False).latest(n)
        # `.latest(n)` returns a single Filing when n == 1 on some edgartools versions.
        if hasattr(filings, "__iter__"):
            return list(filings)
        return [filings]

    return _with_retry(_fetch, desc=f"get_filings({ticker})")


def parse_filing(ticker: str, filing) -> FilingSections:
    """Extracts plain-text sections + XBRL financial facts from a single 10-K Filing."""
    warnings: list[str] = []

    def _load_tenk():
        return filing.obj()

    tenk = _with_retry(_load_tenk, desc=f"filing.obj({ticker}, {filing.accession_no})")

    def _safe_section(attr: str) -> str | None:
        try:
            value = getattr(tenk, attr)
            return str(value) if value else None
        except Exception as e:
            warnings.append(f"{attr}_failed: {e}")
            return None

    business = _safe_section("business")
    risk_factors = _safe_section("risk_factors")
    management_discussion = _safe_section("management_discussion")

    if not risk_factors:
        warnings.append("missing_risk_factors_section")

    try:
        income_statement, balance_sheet, stmt_warnings = extract_statements(tenk)
        warnings.extend(stmt_warnings)
    except Exception as e:
        income_statement = balance_sheet = None
        warnings.append(f"statements_failed: {e}")

    try:
        financials, fact_warnings = extract_financial_facts(filing)
        warnings.extend(fact_warnings)
    except Exception as e:
        from src.data.schema import FinancialFacts

        financials = FinancialFacts()
        warnings.append(f"financial_facts_failed: {e}")

    period = getattr(filing, "period_of_report", None)
    fiscal_year = None
    if period:
        try:
            fiscal_year = int(str(period)[:4])
        except ValueError:
            pass

    return FilingSections(
        ticker=ticker,
        cik=filing.cik,
        company_name=getattr(filing, "company", ticker),
        accession_no=filing.accession_no,
        form_type=filing.form,
        fiscal_year=fiscal_year,
        period_of_report=str(period) if period else None,
        filing_date=str(filing.filing_date) if filing.filing_date else None,
        business=business,
        risk_factors=risk_factors,
        management_discussion=management_discussion,
        income_statement=income_statement,
        balance_sheet=balance_sheet,
        financials=financials,
        parse_warnings=warnings,
    )
