"""Unit tests for src/data/download.py using fake edgartools objects -- no live network calls."""

from __future__ import annotations

import pytest

from src.data.companies import CIK_OVERRIDES, TICKERS
from src.data.download import get_latest_10ks, parse_filing
from src.data.schema import FinancialFacts


class FakeFinancials:
    def __init__(self, metrics):
        self._metrics = metrics

    def get_financial_metrics(self):
        return self._metrics


class FakeStatement:
    """Mimics edgartools' Statement.to_dataframe() shape (see src/data/statements.py)."""

    def __init__(self, rows):
        self._rows = rows

    def to_dataframe(self):
        import pandas as pd

        return pd.DataFrame(self._rows)


def _statement_rows(label: str, value: float):
    return [
        {"label": "ASSETS:", "2025-01-01": None, "abstract": True, "dimension": False},
        {"label": label, "2025-01-01": value, "abstract": False, "dimension": False},
    ]


class FakeTenK:
    def __init__(
        self,
        business,
        risk_factors,
        management_discussion,
        metrics,
        income_statement=None,
        balance_sheet=None,
    ):
        self.business = business
        self.risk_factors = risk_factors
        self.management_discussion = management_discussion
        self.financials = FakeFinancials(metrics)
        self.income_statement = income_statement
        self.balance_sheet = balance_sheet


def fact_row(
    value,
    period_end="2025-01-01",
    period_start="2024-01-02",
    is_dimensioned=False,
    statement_type="IncomeStatement",
    label="Diluted (in dollars per share)",
):
    """One row of `XBRL.query().by_concept(...).to_dataframe()`.

    The real frame carries period, dimension, and statement columns; fact selection depends on all
    three (see src/data/xbrl_facts.py), so the fake has to carry them too.
    """
    return {
        "numeric_value": value,
        "period_end": period_end,
        "period_start": period_start,
        "is_dimensioned": is_dimensioned,
        "statement_type": statement_type,
        "label": label,
    }


class FakeFactQuery:
    def __init__(self, rows):
        self._rows = rows

    def to_dataframe(self):
        import pandas as pd

        return pd.DataFrame(self._rows)


class FakeQueryBuilder:
    def __init__(self, rows_by_concept):
        self._rows_by_concept = rows_by_concept

    def by_concept(self, concept, exact=False):
        if exact:
            return FakeFactQuery(self._rows_by_concept.get(concept, []))
        # The real default matches by prefix, which is what pulled
        # NetIncomeLossAttributableToNoncontrollingInterest into a NetIncomeLoss query.
        rows = [r for c, rs in self._rows_by_concept.items() if c.startswith(concept) for r in rs]
        return FakeFactQuery(rows)


class FakeXBRL:
    def __init__(self, rows_by_concept):
        self._rows_by_concept = rows_by_concept

    def query(self):
        return FakeQueryBuilder(self._rows_by_concept)


class FakeFiling:
    def __init__(self, tenk, xbrl, *, cik=1234, company="Test Co", accession_no="0001-25-000001",
                 form="10-K", period_of_report="2025-01-01", filing_date="2025-02-01"):
        self._tenk = tenk
        self._xbrl = xbrl
        self.cik = cik
        self.company = company
        self.accession_no = accession_no
        self.form = form
        self.period_of_report = period_of_report
        self.filing_date = filing_date

    def obj(self):
        return self._tenk

    def xbrl(self):
        return self._xbrl


def test_parse_filing_happy_path():
    tenk = FakeTenK(
        business="Item 1. Business\nWe make things.",
        risk_factors="Item 1A. Risk Factors\nThings could go wrong.",
        management_discussion="Item 7. MD&A\nRevenue went up.",
        metrics={
            "revenue": 1000.0,
            "net_income": 100.0,
            "total_assets": 5000.0,
            "shares_outstanding_diluted": 50.0,
        },
        income_statement=FakeStatement(_statement_rows("Net sales", 1000.0)),
        balance_sheet=FakeStatement(_statement_rows("Total assets", 5000.0)),
    )
    xbrl = FakeXBRL({
        "us-gaap:Revenues": [fact_row(1000.0, label="Net sales")],
        "us-gaap:NetIncomeLoss": [fact_row(100.0, label="Net income")],
        "us-gaap:Assets": [fact_row(5000.0, statement_type="BalanceSheet", label="Total assets")],
        "us-gaap:EarningsPerShareDiluted": [fact_row(2.0)],
    })
    filing = FakeFiling(tenk, xbrl)

    sections = parse_filing("TEST", filing)

    assert sections.ticker == "TEST"
    assert sections.cik == 1234
    assert sections.fiscal_year == 2025
    assert "make things" in sections.business
    assert "go wrong" in sections.risk_factors
    assert sections.financials.revenue == 1000.0
    assert sections.financials.eps_diluted == 2.0
    assert "Net sales: 1,000" in sections.income_statement
    assert "Total assets: 5,000" in sections.balance_sheet
    assert "ASSETS:" not in sections.income_statement  # abstract header rows are dropped
    assert sections.parse_warnings == []


def test_parse_filing_missing_risk_factors_logs_warning():
    tenk = FakeTenK(
        business="Item 1. Business\nWe make things.",
        risk_factors=None,
        management_discussion="Item 7. MD&A\nRevenue went up.",
        metrics={},
    )
    xbrl = FakeXBRL({})
    filing = FakeFiling(tenk, xbrl)

    sections = parse_filing("TEST", filing)

    assert sections.risk_factors is None
    assert "missing_risk_factors_section" in sections.parse_warnings


def test_parse_filing_derives_eps_when_concept_missing():
    tenk = FakeTenK(
        business="b",
        risk_factors="r",
        management_discussion="m",
        metrics={"net_income": 100.0, "shares_outstanding_diluted": 40.0},
    )
    xbrl = FakeXBRL({})  # no EPS concept available
    filing = FakeFiling(tenk, xbrl)

    sections = parse_filing("TEST", filing)

    assert sections.financials.eps_diluted == pytest.approx(2.5)
    assert "eps_diluted_derived_from_net_income_and_shares" in sections.parse_warnings


# --------------------------------------------------------------------------- ticker resolution


class FakeFilingsList:
    def __init__(self, filings):
        self._filings = filings

    def latest(self, n):
        return self._filings[:n]


def _record_lookups(monkeypatch) -> list:
    """Patches edgar.Company to capture whatever identifier get_latest_10ks resolved to."""
    import edgar

    seen: list = []

    class FakeCompany:
        def __init__(self, cik_or_ticker):
            seen.append(cik_or_ticker)

        def get_filings(self, form, amendments):
            return FakeFilingsList([FakeFiling(None, None)])

    monkeypatch.setattr(edgar, "Company", FakeCompany)
    return seen


def test_a_renamed_ticker_is_resolved_by_cik(monkeypatch):
    """MMC is unresolvable: EDGAR's map holds only the current symbol, and Marsh & McLennan moved
    to MRSH. The CIK survives the rename, so the lookup goes through the number."""
    seen = _record_lookups(monkeypatch)

    get_latest_10ks("MMC")

    assert seen == [62709]


def test_an_ordinary_ticker_is_resolved_by_symbol(monkeypatch):
    seen = _record_lookups(monkeypatch)

    get_latest_10ks("AAPL")

    assert seen == ["AAPL"]


def test_every_cik_override_names_a_company_in_the_universe():
    """An override for a ticker no longer in TICKERS is dead weight that outlives the reason for
    it; one whose CIK is a string would be passed to edgartools as a ticker and fail obscurely."""
    for ticker, cik in CIK_OVERRIDES.items():
        assert ticker in TICKERS, f"{ticker} has a CIK override but is not in the universe"
        assert isinstance(cik, int), f"{ticker} override must be an int CIK, got {type(cik)}"


def test_financial_facts_defaults_are_none():
    facts = FinancialFacts()
    assert facts.revenue is None
    assert facts.unit == "USD"


# --------------------------------------------------------------------------- fact selection
# Each of these reproduces a label defect found in the real corpus. See src/data/xbrl_facts.py.


def _facts_for(rows_by_concept, metrics=None):
    tenk = FakeTenK("b", "r", "m", metrics or {})
    return parse_filing("TEST", FakeFiling(tenk, FakeXBRL(rows_by_concept))).financials


def test_dimensioned_facts_are_skipped():
    """EA's first EPS row is a segment breakdown ('Performance Obligation Period', -0.22); the
    headline 3.51 follows it. Taking row zero labeled 182 filings off the wrong axis."""
    facts = _facts_for({
        "us-gaap:EarningsPerShareDiluted": [
            fact_row(-0.22, is_dimensioned=True, label="Performance Obligation Period"),
            fact_row(3.51),
        ]
    })
    assert facts.eps_diluted == pytest.approx(3.51)


def test_nan_facts_are_treated_as_missing():
    """Wells Fargo's leading EPS rows are text-valued ('Extensible Enumeration') with NaN values.
    float(nan) succeeds, and NaN compares unequal to everything -- so it scores as neither right
    nor wrong, and a NaN label would never show up as an error."""
    facts = _facts_for({
        "us-gaap:EarningsPerShareDiluted": [
            fact_row(float("nan"), label="Extensible Enumeration", statement_type="Disclosures"),
            fact_row(6.26),
        ]
    })
    assert facts.eps_diluted == pytest.approx(6.26)


def test_facts_from_other_periods_are_not_used():
    """A prior year's figure is not this year's answer, even when it is the only one filed."""
    facts = _facts_for({
        "us-gaap:EarningsPerShareDiluted": [fact_row(4.25, period_end="2024-01-01")]
    })
    assert facts.eps_diluted is None


def test_face_of_statement_beats_a_disclosure_note():
    facts = _facts_for({
        "us-gaap:Assets": [
            fact_row(999.0, statement_type="Disclosures", label="Assets, note"),
            fact_row(5000.0, statement_type="BalanceSheet", label="Total assets"),
        ]
    })
    assert facts.total_assets == pytest.approx(5000.0)


def test_xbrl_revenue_wins_over_the_convenience_dict():
    """The dict disagrees with the filed tag for every bank and insurer: Wells Fargo reports 83.7B
    on the face of the income statement, and get_financial_metrics() returns 16.1B."""
    facts = _facts_for(
        {"us-gaap:Revenues": [fact_row(83_699_000_000.0)]},
        metrics={"revenue": 16_081_000_000.0},
    )
    assert facts.revenue == pytest.approx(83_699_000_000.0)


def test_metrics_dict_is_still_used_when_no_concept_is_filed():
    facts = _facts_for({}, metrics={"revenue": 1000.0})
    assert facts.revenue == pytest.approx(1000.0)


def test_concept_match_is_exact_not_a_prefix():
    """AbbVie tags a 7M noncontrolling-interest line under a concept that starts with the one we
    want; a prefix match returns it ahead of the 4,226M net income."""
    facts = _facts_for({
        "us-gaap:NetIncomeLossAttributableToNoncontrollingInterest": [
            fact_row(7_000_000.0, label="Net earnings attributable to noncontrolling interest")
        ],
        "us-gaap:NetIncomeLoss": [
            fact_row(4_226_000_000.0, label="Net earnings attributable to AbbVie Inc.")
        ],
    })
    assert facts.net_income == pytest.approx(4_226_000_000.0)


def test_annual_figure_beats_the_fourth_quarter():
    """A 10-K tags Q4 and the full year against the same period_end."""
    facts = _facts_for({
        "us-gaap:NetIncomeLoss": [
            fact_row(1_816_000_000.0, period_start="2024-10-01", period_end="2025-01-01"),
            fact_row(4_226_000_000.0, period_start="2024-01-02", period_end="2025-01-01"),
        ]
    })
    assert facts.net_income == pytest.approx(4_226_000_000.0)


def test_balance_sheet_instants_are_matched_on_period_instant():
    """Instant facts carry `period_instant` and no `period_end` column at all. Checking only
    `period_end` rejected every balance-sheet fact, sending total_assets to the fallback for all
    182 filings in the corpus."""
    row = {
        "numeric_value": 359_241_000_000.0,
        "period_instant": "2025-01-01",
        "is_dimensioned": False,
        "statement_type": "BalanceSheet",
        "label": "Total assets",
    }
    prior = dict(row, numeric_value=364_980_000_000.0, period_instant="2024-01-01")
    facts = _facts_for({"us-gaap:Assets": [prior, row]})
    assert facts.total_assets == pytest.approx(359_241_000_000.0)
