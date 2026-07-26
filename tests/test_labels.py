"""Unit tests for src/labels/ -- excerpt construction, grounding, splits, and example assembly.

No network calls: the teacher is a fake implementing the `Teacher` protocol.
"""

from __future__ import annotations

import pytest

from src.data.schema import FilingSections, FinancialFacts
from src.labels.build import _risk_section_of, build_example
from src.labels.excerpt import (
    RISK_CHARS,
    build_excerpt,
    check_grounding,
    numeric_variants,
)
from src.labels.schema import ExtractionTarget, RiskFactor
from src.labels.splits import DEFAULT_RATIOS, assign_split, split_counts


def make_sections(**overrides) -> FilingSections:
    defaults = dict(
        ticker="TEST",
        cik=1234,
        company_name="Test Co",
        accession_no="0001-25-000001",
        form_type="10-K",
        fiscal_year=2025,
        business="We make things.",
        risk_factors="Item 1A. Risk Factors\nCompetition could hurt us.",
        management_discussion="Revenue rose.",
        income_statement="## Consolidated Statements of Operations\nNet sales: 1,000",
        balance_sheet="## Consolidated Balance Sheets\nTotal assets: 5,000",
        financials=FinancialFacts(revenue=1000.0, net_income=100.0, total_assets=5000.0),
    )
    defaults.update(overrides)
    return FilingSections(**defaults)


# --- numeric variants / grounding -------------------------------------------------


def test_numeric_variants_covers_filing_scales():
    # 416.161B reported in millions is written "416,161" in a filing.
    variants = numeric_variants(416_161_000_000.0)
    assert "416,161" in variants
    assert "416,161,000,000" in variants


def test_numeric_variants_of_none_is_empty():
    assert numeric_variants(None) == []


def test_check_grounding_flags_missing_figures():
    sections = make_sections(
        income_statement="## Consolidated Statements of Operations\nNet sales: 1,000",
        balance_sheet=None,
    )
    excerpt = build_excerpt(sections)
    grounded = check_grounding(excerpt, sections.financials)

    assert grounded["revenue"] is True
    # total_assets (5,000) appears only on the balance sheet, which is absent here.
    assert grounded["total_assets"] is False


def test_check_grounding_skips_null_facts():
    sections = make_sections(financials=FinancialFacts(revenue=1000.0))
    grounded = check_grounding(build_excerpt(sections), sections.financials)
    assert "net_income" not in grounded


# --- excerpt ----------------------------------------------------------------------


def test_excerpt_includes_statements_before_narrative():
    """Statements carry the numeric answers, so they must survive any downstream truncation."""
    excerpt = build_excerpt(make_sections())
    assert excerpt.index("Consolidated Statements of Operations") < excerpt.index("Item 1A")


def test_excerpt_truncates_long_risk_factors():
    sections = make_sections(risk_factors="Item 1A. Risk Factors\n" + ("risk. " * 20_000))
    excerpt = build_excerpt(sections)
    risk_part = excerpt.split("## Item 1A. Risk Factors\n", 1)[1]
    assert len(risk_part) <= RISK_CHARS


def test_excerpt_windows_mdna_around_figures():
    """MD&A is windowed on the figures, not truncated from the top."""
    filler = "boilerplate. " * 3_000
    sections = make_sections(
        management_discussion=filler + " Total net sales were 1,000 this year. " + filler,
        financials=FinancialFacts(revenue=1000.0),
    )
    excerpt = build_excerpt(sections)
    assert "Total net sales were 1,000" in excerpt


def test_excerpt_handles_missing_sections():
    sections = make_sections(business=None, management_discussion=None, risk_factors=None)
    excerpt = build_excerpt(sections)
    assert "Test Co" in excerpt
    assert "Item 1A" not in excerpt


# --- splits -----------------------------------------------------------------------


def test_assign_split_is_deterministic_and_case_insensitive():
    assert assign_split("AAPL") == assign_split("AAPL") == assign_split("aapl")


def test_split_ratios_are_roughly_respected():
    tickers = [f"TICK{i}" for i in range(2_000)]
    counts = split_counts(tickers)
    assert counts["train"] + counts["val"] + counts["test"] == 2_000
    # Hash-bucketed, so allow slack; this catches a broken assignment, not sampling noise.
    assert abs(counts["train"] / 2_000 - DEFAULT_RATIOS["train"]) < 0.05


def test_no_company_appears_in_two_splits():
    """The anti-leakage guarantee: one company resolves to exactly one split."""
    tickers = [f"TICK{i}" for i in range(500)]
    buckets = {"train": set(), "val": set(), "test": set()}
    for t in tickers:
        buckets[assign_split(t)].add(t)
    assert not buckets["train"] & buckets["val"]
    assert not buckets["train"] & buckets["test"]
    assert not buckets["val"] & buckets["test"]


# --- example assembly -------------------------------------------------------------


def test_risk_section_of_returns_only_item_1a():
    excerpt = build_excerpt(make_sections())
    risk_section = _risk_section_of(excerpt)
    assert risk_section.startswith("## Item 1A. Risk Factors")
    assert "Item 1. Business" not in risk_section


def test_risk_section_of_missing_section_is_empty():
    excerpt = build_excerpt(make_sections(risk_factors=None))
    assert _risk_section_of(excerpt) == ""


def test_build_example_shape_and_valid_target():
    risks = [RiskFactor(title="Competition", category="competitive", summary="Rivals may win.")]
    example = build_example(make_sections(), risks)

    assert [m["role"] for m in example["messages"]] == ["system", "user", "assistant"]
    assert example["ticker"] == "TEST"

    # The assistant turn must be valid, parseable target JSON -- it is the training signal.
    target = ExtractionTarget.model_validate_json(example["messages"][2]["content"])
    assert target.financials.revenue == 1000.0
    assert target.risk_factors[0].category == "competitive"


def test_build_example_labels_are_present_in_the_input():
    """Every labeled risk must be supported by the excerpt the model sees."""
    example = build_example(make_sections(), [])
    user_turn = example["messages"][1]["content"]
    assert "Competition could hurt us." in user_turn


# --- failure handling -------------------------------------------------------------


class FailingTeacher:
    """Teacher whose every call fails, as an exhausted-credit account does."""

    def __init__(self, exc):
        self._exc = exc
        self.calls = 0

    def label_risk_factors(self, risk_text: str):
        self.calls += 1
        raise self._exc


def test_unavailable_teacher_aborts_after_one_call(tmp_path, monkeypatch):
    """A permanent failure must stop the run, not retry it once per filing."""
    from src.labels import build
    from src.labels.teacher import TeacherUnavailableError

    monkeypatch.setattr(build, "RAW_DIR", tmp_path / "raw")
    monkeypatch.setattr(build, "PROCESSED_DIR", tmp_path / "processed")
    build.RAW_DIR.mkdir(parents=True)
    for i in range(3):
        sections = make_sections(ticker=f"TICK{i}", accession_no=f"0001-25-00000{i}")
        (build.RAW_DIR / f"{i}.json").write_text(sections.model_dump_json())

    teacher = FailingTeacher(TeacherUnavailableError("credit balance too low"))
    with pytest.raises(SystemExit):
        build.run(teacher=teacher)

    assert teacher.calls == 1, "should abort on the first permanent failure, not retry per filing"


def test_failed_run_does_not_clobber_existing_dataset(tmp_path, monkeypatch):
    """The regression that motivated the guard: a billing outage emptied train.jsonl."""
    from src.labels import build
    from src.labels.teacher import TeacherError

    monkeypatch.setattr(build, "RAW_DIR", tmp_path / "raw")
    monkeypatch.setattr(build, "PROCESSED_DIR", tmp_path / "processed")
    build.RAW_DIR.mkdir(parents=True)
    build.PROCESSED_DIR.mkdir(parents=True)

    (build.RAW_DIR / "a.json").write_text(make_sections().model_dump_json())
    good = build.PROCESSED_DIR / "train.jsonl"
    good.write_text('{"existing": "data"}\n')

    with pytest.raises(SystemExit):
        build.run(teacher=FailingTeacher(TeacherError("transient")))

    assert good.read_text() == '{"existing": "data"}\n'


def test_allow_partial_writes_despite_failures(tmp_path, monkeypatch):
    from src.labels import build
    from src.labels.teacher import TeacherError

    monkeypatch.setattr(build, "RAW_DIR", tmp_path / "raw")
    monkeypatch.setattr(build, "PROCESSED_DIR", tmp_path / "processed")
    build.RAW_DIR.mkdir(parents=True)
    (build.RAW_DIR / "a.json").write_text(make_sections().model_dump_json())

    build.run(teacher=FailingTeacher(TeacherError("transient")), allow_partial=True)
    assert (build.PROCESSED_DIR / "train.jsonl").exists()


# --- incorporation by reference ---------------------------------------------------


def test_detects_risk_factors_incorporated_by_reference():
    """WFC and USB satisfy Item 1A with a pointer to their Annual Report, not risk text."""
    from src.labels.build import is_incorporated_by_reference

    wfc = (
        "ITEM 1A. RISK FACTORS\nInformation in response to this Item 1A can be found in this "
        "report under Item 1 and in the 2025 Annual Report to Shareholders."
    )
    assert is_incorporated_by_reference(wfc) is True


def test_real_risk_section_is_not_flagged_as_by_reference():
    from src.labels.build import is_incorporated_by_reference

    real = "Item 1A. Risk Factors\n" + ("We face substantial competition in every market. " * 100)
    assert is_incorporated_by_reference(real) is False
    assert is_incorporated_by_reference(None) is False


def test_example_carries_the_by_reference_flag():
    from src.labels.build import build_example

    flagged = build_example(
        make_sections(risk_factors="Item 1A. Risk Factors\nSee the Annual Report; incorporated."),
        [],
    )
    assert flagged["risk_factors_by_reference"] is True
    assert build_example(make_sections(), [])["risk_factors_by_reference"] is False


def test_ungrounded_numeric_labels_are_dropped():
    """A figure absent from the excerpt cannot be derived from it; keeping it teaches the model to
    invent numbers. Truist files no total-revenue tag, and the fallback's 471M reached the labels."""
    from src.data.schema import FinancialFacts
    from src.labels.build import grounded_financials

    excerpt = "Total assets: 5,000,000\nNet income: 250,000"
    facts = FinancialFacts(revenue=471_000_000.0, net_income=250_000.0, total_assets=5_000_000.0)
    kept = grounded_financials(excerpt, facts)

    assert kept.revenue is None  # not in the excerpt
    assert kept.net_income == 250_000.0
    assert kept.total_assets == 5_000_000.0
