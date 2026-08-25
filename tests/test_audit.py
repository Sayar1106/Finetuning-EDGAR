"""Unit tests for src/labels/audit.py -- sampling, staleness detection, and clustered scoring.

No network and no interactive input: `review` is exercised only through the pure functions it
writes to, so the tests never block on stdin.
"""

from __future__ import annotations

import json

from src.data.schema import FinancialFacts
from src.labels.audit import (
    LabeledFiling,
    blind_category,
    check_notes,
    draw_sample,
    format_report,
    is_uninformative,
    load_labeled_filings,
    risk_section_of,
    score_sheet,
)
from src.labels.schema import ExtractionTarget, RiskFactor


def make_risks(n: int, category: str = "market") -> list[RiskFactor]:
    return [
        RiskFactor(title=f"Risk {i}", category=category, summary=f"Summary of risk {i}.")
        for i in range(n)
    ]


def make_filing(ticker: str = "TEST", n_risks: int = 6, **overrides) -> LabeledFiling:
    defaults = dict(
        ticker=ticker,
        accession_no=f"0001-25-{ticker}",
        split="train",
        risk_text="## Item 1A. Risk Factors\nThings could go wrong.",
        risks=make_risks(n_risks),
    )
    defaults.update(overrides)
    return LabeledFiling(**defaults)


def write_dataset(tmp_path, rows_by_split: dict[str, list[dict]]):
    for split, rows in rows_by_split.items():
        path = tmp_path / f"{split}.jsonl"
        with path.open("w") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")
    return tmp_path


def make_row(ticker: str, n_risks: int = 3, by_reference: bool = False) -> dict:
    target = ExtractionTarget(
        company=f"{ticker} Inc",
        ticker=ticker,
        fiscal_year=2025,
        form_type="10-K",
        financials=FinancialFacts(revenue=1.0),
        risk_factors=make_risks(n_risks),
    )
    return {
        "ticker": ticker,
        "accession_no": f"0001-25-{ticker}",
        "risk_factors_by_reference": by_reference,
        "messages": [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "## Item 1. Business\nstuff\n## Item 1A. Risk Factors\nrisky"},
            {"role": "assistant", "content": target.model_dump_json()},
        ],
    }


def make_sheet_row(idx: int, teacher: str, human: str | None, **overrides) -> dict:
    row = {
        "id": f"acc#{idx}",
        "ticker": "TEST",
        "accession_no": "acc",
        "split": "train",
        "risk_index": idx,
        "risk_digest": "abc123",
        "teacher_title": f"Risk {idx}",
        "teacher_category": teacher,
        "teacher_summary": "s",
        "human_category": human,
        "grounded": True,
        "summary_ok": True,
        "note": "",
    }
    row.update(overrides)
    return row


# --- loading ----------------------------------------------------------------------


def test_risk_section_starts_at_the_item_1a_marker():
    excerpt = "## Item 1. Business\nWe sell things.\n## Item 1A. Risk Factors\nCompetition."
    assert risk_section_of(excerpt).startswith("## Item 1A. Risk Factors")
    assert "We sell things" not in risk_section_of(excerpt)


def test_risk_section_is_empty_when_the_marker_is_absent():
    assert risk_section_of("## Item 1. Business\nno risks here") == ""


def test_by_reference_filings_are_excluded_from_the_population(tmp_path):
    # Their empty risk list is correct by construction, so auditing it would pad the denominator
    # with items that cannot disagree.
    write_dataset(
        tmp_path,
        {"train": [make_row("AAA"), make_row("BBB", n_risks=0, by_reference=True)]},
    )
    filings = load_labeled_filings(tmp_path)
    assert [f.ticker for f in filings] == ["AAA"]


def test_the_loaded_risk_text_is_the_slice_the_teacher_saw(tmp_path):
    write_dataset(tmp_path, {"train": [make_row("AAA")]})
    (filing,) = load_labeled_filings(tmp_path)
    assert filing.risk_text.startswith("## Item 1A. Risk Factors")


def test_a_missing_split_file_is_not_an_error(tmp_path):
    write_dataset(tmp_path, {"train": [make_row("AAA")]})
    assert len(load_labeled_filings(tmp_path)) == 1


# --- sampling ---------------------------------------------------------------------


def test_the_sample_is_reproducible_from_its_seed():
    filings = [make_filing(f"T{i}") for i in range(20)]
    first, _ = draw_sample(filings, n_filings=5, per_filing=3, seed=7)
    second, _ = draw_sample(filings, n_filings=5, per_filing=3, seed=7)
    assert [r["id"] for r in first] == [r["id"] for r in second]


def test_a_different_seed_draws_a_different_sample():
    filings = [make_filing(f"T{i}") for i in range(20)]
    a, _ = draw_sample(filings, n_filings=5, per_filing=3, seed=0)
    b, _ = draw_sample(filings, n_filings=5, per_filing=3, seed=1)
    assert [r["id"] for r in a] != [r["id"] for r in b]


def test_a_filing_with_fewer_risks_than_the_quota_contributes_all_of_them():
    filings = [make_filing("SMALL", n_risks=2)]
    risks, filing_rows = draw_sample(filings, n_filings=1, per_filing=4, seed=0)
    assert len(risks) == 2
    assert filing_rows[0]["n_sampled"] == 2
    assert filing_rows[0]["n_teacher_risks"] == 2


def test_sampled_rows_start_unreviewed():
    risks, filing_rows = draw_sample([make_filing()], n_filings=1, per_filing=2, seed=0)
    assert all(r["human_category"] is None for r in risks)
    assert all(r["grounded"] is None for r in risks)
    assert filing_rows[0]["missed_risks"] is None


def test_the_test_stratum_is_drawn_only_from_test_split_filings():
    filings = [make_filing(f"TR{i}") for i in range(20)]
    filings += [make_filing(f"TE{i}", split="test") for i in range(5)]
    _, filing_rows = draw_sample(filings, n_filings=4, per_filing=2, seed=0, test_filings=2)
    extra = [f for f in filing_rows if f["stratum"] == "test"]
    assert len(extra) == 2
    assert all(f["split"] == "test" for f in extra)


def test_a_filing_is_never_reviewed_in_both_strata():
    # Double-counting a filing would put the same verdicts in the unbiased corpus estimate and in
    # the test-split check, correlating two numbers that are reported as independent.
    filings = [make_filing(f"TE{i}", split="test") for i in range(6)]
    _, filing_rows = draw_sample(filings, n_filings=3, per_filing=2, seed=0, test_filings=3)
    accessions = [f["accession_no"] for f in filing_rows]
    assert len(accessions) == len(set(accessions)) == 6


def test_the_test_stratum_is_skipped_when_asked_for_zero():
    filings = [make_filing(f"TE{i}", split="test") for i in range(6)]
    _, filing_rows = draw_sample(filings, n_filings=2, per_filing=2, seed=0, test_filings=0)
    assert {f["stratum"] for f in filing_rows} == {"random"}


def test_scoring_one_stratum_ignores_rows_from_the_other():
    rows = [
        make_sheet_row(0, "market", "market", stratum="random"),
        make_sheet_row(1, "cyber", "legal", stratum="test"),
    ]
    assert score_sheet(rows, [], n_resamples=0, stratum="random").category_agreement == 1.0
    assert score_sheet(rows, [], n_resamples=0, stratum="test").category_agreement == 0.0


def test_the_digest_changes_when_the_labels_change():
    before = make_filing().risk_digest
    after = make_filing(risks=make_risks(6, category="cyber")).risk_digest
    assert before != after


# --- scoring ----------------------------------------------------------------------


def test_unreviewed_rows_are_excluded_rather_than_counted_as_agreement():
    rows = [
        make_sheet_row(0, "market", "market"),
        make_sheet_row(1, "cyber", None),
    ]
    report = score_sheet(rows, [], n_resamples=0)
    assert report.n_reviewed == 1
    assert report.n_total == 2
    assert report.category_agreement == 1.0


def test_category_agreement_counts_exact_matches():
    rows = [
        make_sheet_row(0, "market", "market"),
        make_sheet_row(1, "cyber", "cyber"),
        make_sheet_row(2, "legal", "regulatory"),
        make_sheet_row(3, "talent", "talent"),
    ]
    report = score_sheet(rows, [], n_resamples=0)
    assert report.category_agreement == 0.75
    assert report.confusions == {("legal", "regulatory"): 1}


def test_grounding_and_summary_verdicts_score_independently():
    rows = [
        make_sheet_row(0, "market", "market", grounded=False, summary_ok=True),
        make_sheet_row(1, "cyber", "cyber", grounded=True, summary_ok=False),
    ]
    report = score_sheet(rows, [], n_resamples=0)
    assert report.category_agreement == 1.0
    assert report.grounded_rate == 0.5
    assert report.summary_ok_rate == 0.5


def test_rows_are_dropped_when_the_labels_they_judged_no_longer_exist():
    # The sheet was drawn against one set of labels; the dataset has since been rebuilt. Scoring
    # those verdicts against the new labels would attribute a human judgement to a sentence the
    # human never read.
    rows = [make_sheet_row(0, "market", "market", accession_no="acc", risk_digest="OLD")]
    filings = [make_filing(accession_no="acc")]
    report = score_sheet(rows, [], filings=filings, n_resamples=0)
    assert report.stale == ["acc#0"]
    assert report.n_reviewed == 0


def test_a_matching_digest_is_not_stale():
    filing = make_filing(accession_no="acc")
    rows = [make_sheet_row(0, "market", "market", risk_digest=filing.risk_digest)]
    report = score_sheet(rows, [], filings=[filing], n_resamples=0)
    assert report.stale == []
    assert report.n_reviewed == 1


def test_implied_recall_uses_missed_risks_against_labeled_risks():
    filing_rows = [
        {"accession_no": "acc", "n_teacher_risks": 9, "missed_risks": 1},
        {"accession_no": "acc2", "n_teacher_risks": 9, "missed_risks": 1},
    ]
    report = score_sheet([], filing_rows, n_resamples=0)
    assert report.missed_risks == 2
    assert report.implied_recall == 18 / 20


def test_recall_is_none_when_no_filing_was_read_end_to_end():
    filing_rows = [{"accession_no": "acc", "n_teacher_risks": 9, "missed_risks": None}]
    report = score_sheet([], filing_rows, n_resamples=0)
    assert report.missed_risks is None
    assert report.implied_recall is None


# --- clustered intervals ----------------------------------------------------------


def test_the_interval_brackets_the_point_estimate():
    rows = [make_sheet_row(i, "market", "market" if i % 4 else "cyber") for i in range(20)]
    for i, row in enumerate(rows):
        row["accession_no"] = f"acc{i // 4}"
    report = score_sheet(rows, [], n_resamples=500)
    ci = report.cis["category_agreement"]
    assert ci.low <= report.category_agreement <= ci.high


def test_clustering_by_filing_widens_the_interval_versus_treating_items_as_independent():
    # Same 20 verdicts, same 25% disagreement rate. In `clustered` every disagreement sits in one
    # filing, so a resample either takes that filing or does not; in `spread` they are scattered
    # across filings, so no single draw can swing the rate as far.
    clustered = [make_sheet_row(i, "market", "market") for i in range(20)]
    for i, row in enumerate(clustered):
        row["accession_no"] = f"acc{i // 4}"
        if i < 5:
            row["human_category"] = "cyber"

    spread = [make_sheet_row(i, "market", "market") for i in range(20)]
    for i, row in enumerate(spread):
        row["accession_no"] = f"acc{i // 4}"
        if i % 4 == 0:
            row["human_category"] = "cyber"

    a = score_sheet(clustered, [], n_resamples=2000).cis["category_agreement"]
    b = score_sheet(spread, [], n_resamples=2000).cis["category_agreement"]
    assert (a.high - a.low) > (b.high - b.low)


def test_no_interval_is_produced_from_an_empty_sheet():
    report = score_sheet([], [], n_resamples=100)
    assert report.cis == {}
    assert report.category_agreement is None


def test_no_interval_is_produced_from_too_few_filings():
    # One cluster makes every resample identical, so the bootstrap would print a point estimate
    # dressed as a 95% interval -- worse than reporting no interval at all.
    rows = [make_sheet_row(i, "market", "market") for i in range(4)]
    report = score_sheet(rows, [], n_resamples=500)
    assert report.category_agreement == 1.0
    assert report.cis == {}
    assert "under the" in format_report(report)


# --- note markers: blind verdicts and uninformative rows --------------------------


REVISED = "revised 2026-01-01: blind verdict was '{}'; changed to match teacher. NOT a blind verdict."
UNINFORMATIVE = "UNINFORMATIVE: no recoverable rule for this theme; exclude when reporting."


def test_the_published_rate_is_blind_not_the_revised_answer():
    # Both rows read as agreement on the sheet, but one was changed after seeing the teacher's.
    rows = [
        make_sheet_row(0, "market", "market"),
        make_sheet_row(1, "operational", "operational", note=REVISED.format("cyber")),
    ]
    report = score_sheet(rows, [], n_resamples=0)
    assert report.category_agreement == 0.5
    assert report.category_agreement_recorded == 1.0
    assert report.n_revised == 1


def test_a_revision_that_still_disagrees_counts_against_both_rates():
    rows = [make_sheet_row(0, "legal", "regulatory", note=REVISED.format("climate"))]
    report = score_sheet(rows, [], n_resamples=0)
    assert report.category_agreement == 0.0
    assert report.category_agreement_recorded == 0.0


def test_a_keying_correction_scores_as_a_blind_match():
    # `corrected` carries no "blind verdict was" clause: the reviewer's judgement never changed,
    # so human_category *is* the blind verdict and the row is a genuine match.
    note = "corrected 2026-01-01: keying error. Recorded as 'cyber'; intended 'financial'."
    rows = [make_sheet_row(0, "financial", "financial", note=note)]
    report = score_sheet(rows, [], n_resamples=0)
    assert report.category_agreement == 1.0
    assert report.n_revised == 0


def test_confusions_report_the_blind_disagreement_not_the_revised_one():
    rows = [make_sheet_row(0, "operational", "operational", note=REVISED.format("cyber"))]
    report = score_sheet(rows, [], n_resamples=0)
    assert report.confusions == {("operational", "cyber"): 1}


def test_uninformative_rows_drop_out_of_the_category_rate():
    rows = [
        make_sheet_row(0, "market", "market"),
        make_sheet_row(1, "macroeconomic", "market", note=UNINFORMATIVE),
    ]
    report = score_sheet(rows, [], n_resamples=0)
    assert report.n_reviewed == 2
    assert report.n_uninformative == 1
    assert report.category_agreement == 1.0  # the flagged disagreement is excluded, not counted
    assert report.confusions == {}


def test_uninformative_rows_still_score_grounding_and_summary():
    # Only the category was unanswerable; the reviewer could still read the text.
    rows = [
        make_sheet_row(0, "market", "market", grounded=True, summary_ok=True),
        make_sheet_row(
            1, "macroeconomic", "market", grounded=False, summary_ok=False, note=UNINFORMATIVE
        ),
    ]
    report = score_sheet(rows, [], n_resamples=0)
    assert report.grounded_rate == 0.5
    assert report.summary_ok_rate == 0.5


def test_the_word_unflagged_is_not_an_uninformative_marker():
    row = make_sheet_row(0, "market", "market", note="unflagged 2026-01-01: not UNINFORMATIVE after all")
    assert is_uninformative(row) is False


def test_blind_category_falls_back_to_the_recorded_answer():
    assert blind_category(make_sheet_row(0, "market", "market")) == "market"
    assert blind_category(make_sheet_row(0, "market", None)) is None


def test_check_notes_catches_a_revision_with_no_recoverable_verdict():
    rows = [make_sheet_row(0, "market", "market", note="revised 2026-01-01: changed my mind")]
    problems = check_notes(rows)
    assert len(problems) == 1
    assert "blind verdict" in problems[0]


def test_check_notes_catches_an_unparseable_uninformative_mention():
    rows = [make_sheet_row(0, "market", "market", note="this looks UNINFORMATIVE to me")]
    problems = check_notes(rows)
    assert len(problems) == 1
    assert "still being scored" in problems[0]


def test_check_notes_is_silent_on_well_formed_markers():
    rows = [
        make_sheet_row(0, "market", "market"),
        make_sheet_row(1, "operational", "operational", note=REVISED.format("cyber")),
        make_sheet_row(2, "macroeconomic", "market", note=UNINFORMATIVE),
    ]
    assert check_notes(rows) == []


def test_the_anchoring_gap_interval_is_reported_when_rows_were_revised():
    rows = [
        make_sheet_row(i, "market", "market", accession_no=f"acc{i // 2}", id=f"acc{i // 2}#{i}")
        for i in range(6)
    ]
    rows[0]["note"] = REVISED.format("cyber")
    report = score_sheet(rows, [], n_resamples=200)
    gap = report.cis["anchoring_gap"]
    assert gap.low >= 0.0
    assert gap.high > 0.0
    assert "Anchoring gap" in format_report(report)
