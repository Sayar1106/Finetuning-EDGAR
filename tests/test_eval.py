"""Unit tests for src/eval/metrics.py. Pure functions -- no model, no network."""

from __future__ import annotations

import json

import pytest

from src.data.schema import FinancialFacts
from src.eval.metrics import (
    CLOSE,
    EXACT,
    MISSING,
    NOT_SCORED,
    SCALE_ERROR,
    SPURIOUS,
    WRONG,
    extract_json,
    format_report,
    is_scale_error,
    match_risks,
    parse_prediction,
    report_to_dict,
    risk_similarity,
    score_dataset,
    score_example,
    score_field,
    score_numerics,
    score_risks,
)
from src.labels.schema import ExtractionTarget, RiskFactor

CYBER = RiskFactor(
    title="Cybersecurity breaches could expose customer data",
    category="cyber",
    summary="A successful intrusion would expose customer records and trigger notification costs.",
)
CYBER_PARAPHRASE = RiskFactor(
    title="Cyber intrusions may expose customer data",
    category="cyber",
    summary="An intrusion would expose customer records and lead to notification expenses.",
)
CYBER_MISCATEGORIZED = RiskFactor(
    title="Cyber intrusions may expose customer data",
    category="operational",
    summary="An intrusion would expose customer records and lead to notification expenses.",
)
FX = RiskFactor(
    title="Foreign currency exchange rate fluctuations",
    category="market",
    summary="Revenue earned abroad converts unfavorably when the dollar strengthens.",
)


def make_target(**overrides) -> ExtractionTarget:
    defaults = dict(
        company="Test Co",
        ticker="TEST",
        fiscal_year=2025,
        form_type="10-K",
        financials=FinancialFacts(
            revenue=1_000_000_000.0, net_income=50_000_000.0, total_assets=2_000_000_000.0, eps_diluted=2.50
        ),
        risk_factors=[CYBER, FX],
    )
    defaults.update(overrides)
    return ExtractionTarget(**defaults)


def make_example(target: ExtractionTarget | None = None, by_reference: bool = False) -> dict:
    target = target or make_target()
    return {
        "ticker": target.ticker,
        "accession_no": "0001-25-000001",
        "risk_factors_by_reference": by_reference,
        "messages": [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "excerpt"},
            {"role": "assistant", "content": target.model_dump_json()},
        ],
    }


# --------------------------------------------------------------------------- JSON extraction


def test_extract_json_from_fenced_block():
    raw = 'Here you go:\n```json\n{"a": 1}\n```\nHope that helps!'
    assert json.loads(extract_json(raw)) == {"a": 1}


def test_extract_json_from_surrounding_prose():
    assert json.loads(extract_json('Sure. {"a": {"b": 2}} Let me know.')) == {"a": {"b": 2}}


def test_extract_json_ignores_braces_inside_strings():
    raw = '{"summary": "a } brace in prose", "n": 1}'
    assert json.loads(extract_json(raw))["n"] == 1


def test_extract_json_returns_none_without_an_object():
    assert extract_json("I cannot help with that.") is None
    assert extract_json("") is None


def test_strict_validity_requires_a_bare_json_object():
    target = make_target()
    clean = parse_prediction(target.model_dump_json())
    assert clean.strict_valid and clean.lenient_valid

    wrapped = parse_prediction(f"Certainly!\n```json\n{target.model_dump_json()}\n```")
    assert not wrapped.strict_valid
    assert wrapped.lenient_valid
    assert wrapped.target.ticker == "TEST"


def test_parse_prediction_rejects_wrong_shape():
    parsed = parse_prediction('{"company": "Test Co"}')  # missing required form_type/ticker
    assert not parsed.usable
    assert "schema" in parsed.error


def test_parse_prediction_rejects_unknown_risk_category():
    bad = json.dumps(
        {
            "company": "Test Co",
            "ticker": "TEST",
            "form_type": "10-K",
            "financials": {},
            "risk_factors": [{"title": "t", "category": "vibes", "summary": "s"}],
        }
    )
    assert not parse_prediction(bad).usable


# --------------------------------------------------------------------------- numeric scoring


def test_score_field_verdicts():
    assert score_field(1000.0, 1000.0) == EXACT
    assert score_field(22.18, 22.2) == CLOSE  # filer-rounded
    assert score_field(1000.0, 1500.0) == WRONG
    assert score_field(1000.0, None) == MISSING
    assert score_field(None, 1000.0) == SPURIOUS
    assert score_field(None, None) == NOT_SCORED


def test_scale_errors_are_distinguished_from_wrong_answers():
    # The model copied "233,593" from a statement printed in millions.
    assert score_field(233_593_000_000.0, 233_593.0) == SCALE_ERROR
    assert score_field(233_593_000_000.0, 233.593) == SCALE_ERROR  # billions
    assert is_scale_error(-5_000_000.0, -5_000.0)


def test_sign_flip_is_wrong_not_a_scale_error():
    """A loss reported as a profit is a correctness failure, not a units failure."""
    assert not is_scale_error(-5_000_000.0, 5_000.0)
    assert score_field(-5_000_000.0, 5_000.0) == WRONG


def test_zero_gold_does_not_divide_by_zero():
    assert score_field(0.0, 0.0) == EXACT
    assert score_field(0.0, 100.0) == WRONG
    assert not is_scale_error(0.0, 100.0)


def test_score_numerics_treats_invalid_output_as_all_missing():
    verdicts = score_numerics(make_target().financials, None)
    assert set(verdicts.values()) == {MISSING}


# --------------------------------------------------------------------------- risk scoring


def test_paraphrases_match_and_unrelated_risks_do_not():
    assert risk_similarity(CYBER, CYBER_PARAPHRASE) > 0.30
    assert risk_similarity(CYBER, FX) == 0.0


def test_match_risks_is_one_to_one():
    """Two predictions describing the same gold risk must not both count as hits."""
    matches = match_risks([CYBER], [CYBER, CYBER_PARAPHRASE])
    assert len(matches) == 1


def test_perfect_prediction_scores_one():
    score = score_risks([CYBER, FX], [CYBER, FX])
    assert score.category_f1 == pytest.approx(1.0)
    assert score.match_f1 == pytest.approx(1.0)
    assert score.category_accuracy == pytest.approx(1.0)


def test_category_f1_ignores_wording_but_match_f1_does_not():
    """The two risk metrics fail differently, which is the point of reporting both."""
    score = score_risks([CYBER], [CYBER_PARAPHRASE])
    assert score.category_f1 == pytest.approx(1.0)  # both are one `cyber` risk
    assert score.match_f1 == pytest.approx(1.0)  # wording is close enough to match

    unrelated = score_risks([CYBER], [FX])
    assert unrelated.category_f1 == pytest.approx(0.0)
    assert unrelated.match_f1 == pytest.approx(0.0)


def test_matched_risk_with_wrong_category_lowers_category_accuracy():
    score = score_risks([CYBER], [CYBER_MISCATEGORIZED])
    assert score.matched == 1
    assert score.matched_same_category == 0
    assert score.category_accuracy == pytest.approx(0.0)
    assert score.category_f1 == pytest.approx(0.0)  # cyber vs operational: no multiset overlap


def test_over_prediction_costs_precision():
    score = score_risks([CYBER], [CYBER, FX])
    assert score.match_f1 == pytest.approx(2 * 1.0 * 0.5 / 1.5)  # P=0.5, R=1.0


def test_empty_prediction_scores_zero_without_error():
    score = score_risks([CYBER, FX], [])
    assert score.match_f1 == 0.0
    assert score.category_accuracy is None  # nothing matched -- undefined, not zero


# --------------------------------------------------------------------------- example / corpus


def test_score_example_perfect_round_trip():
    target = make_target()
    score = score_example(make_example(target), target.model_dump_json())
    assert score.strict_valid
    assert set(score.numerics.values()) == {EXACT}
    assert score.risk.match_f1 == pytest.approx(1.0)


def test_by_reference_filing_skips_risk_scoring_but_keeps_numerics():
    """WFC's Item 1A is a cross-reference; predicting no risks is correct, not a miss."""
    target = make_target(risk_factors=[])
    example = make_example(target, by_reference=True)
    score = score_example(example, target.model_dump_json())
    assert score.risk is None
    assert set(score.numerics.values()) == {EXACT}


def test_invalid_output_is_scored_as_a_miss_not_excluded():
    example = make_example()
    report = score_dataset([example], ["I'm sorry, I can't read that filing."])
    assert report.strict_valid == 0 and report.lenient_valid == 0
    assert report.numeric_accuracy == pytest.approx(0.0)
    assert report.macro_match_f1 == pytest.approx(0.0)
    assert report.n_risk_scored == 1  # counted, and counted as a failure


def test_macro_average_is_not_dominated_by_the_most_verbose_filer():
    """One filing discloses 46 risks and another 5; per-filing averaging keeps them equal."""
    verbose_gold = make_target(risk_factors=[CYBER] * 40 + [FX])
    terse_gold = make_target(risk_factors=[FX])

    examples = [make_example(verbose_gold), make_example(terse_gold)]
    predictions = [
        verbose_gold.model_dump_json(),  # perfect on the verbose filing
        make_target(risk_factors=[CYBER]).model_dump_json(),  # entirely wrong on the terse one
    ]
    report = score_dataset(examples, predictions)

    assert report.macro_match_f1 == pytest.approx(0.5)  # (1.0 + 0.0) / 2
    assert report.risk_micro.match_f1 > 0.9  # micro would have called this near-perfect


def test_report_serializes_and_formats():
    target = make_target()
    report = score_dataset([make_example(target)], [target.model_dump_json()], model="test-model")

    payload = report_to_dict(report)
    assert payload["model"] == "test-model"
    assert payload["strict_valid_rate"] == pytest.approx(1.0)
    assert payload["numeric_by_field"]["revenue"][EXACT] == 1
    assert payload["per_filing"][0]["ticker"] == "TEST"
    json.dumps(payload)  # must be JSON-serializable for data/eval/

    text = format_report(report)
    assert "test-model" in text and "Numeric accuracy" in text


def test_score_dataset_rejects_length_mismatch():
    with pytest.raises(ValueError):
        score_dataset([make_example()], [])


# --------------------------------------------------------------------------------------------
# Bootstrap confidence intervals
# --------------------------------------------------------------------------------------------


def _mixed_dataset(n_correct: int, n_wrong: int):
    """`n_correct` perfect filings and `n_wrong` entirely wrong ones."""
    target = make_target()
    wrong = make_target(financials=FinancialFacts(revenue=1.0, net_income=1.0, total_assets=1.0,
                                                  eps_diluted=1.0))
    examples = [make_example(target) for _ in range(n_correct + n_wrong)]
    predictions = [target.model_dump_json()] * n_correct + [wrong.model_dump_json()] * n_wrong
    return examples, predictions


def test_interval_brackets_the_point_estimate():
    report = score_dataset(*_mixed_dataset(8, 4), bootstrap=2000)

    ci = report.cis["numeric_accuracy"]
    assert ci.low <= report.numeric_accuracy <= ci.high
    assert ci.confidence == pytest.approx(0.95)


def test_a_larger_test_set_gives_a_narrower_interval():
    """The whole point of reporting these: n=12 cannot support the comparisons n=120 can."""
    small = score_dataset(*_mixed_dataset(10, 2), bootstrap=2000)
    large = score_dataset(*_mixed_dataset(100, 20), bootstrap=2000)

    small_width = small.cis["numeric_accuracy"].high - small.cis["numeric_accuracy"].low
    large_width = large.cis["numeric_accuracy"].high - large.cis["numeric_accuracy"].low
    assert large_width < small_width / 2


def test_resampling_is_over_filings_not_field_instances():
    """Four numeric fields per filing are correlated -- a filing is right or wrong on all of them
    here. Resampling instances would report an interval ~2x too narrow."""
    report = score_dataset(*_mixed_dataset(6, 6), bootstrap=4000)

    ci = report.cis["numeric_accuracy"]
    width = ci.high - ci.low
    # Filing-level SE at p=0.5, n=12 is ~0.144 -> ~0.57 wide. Instance-level (n=48) would be ~0.28.
    assert width > 0.4


def test_intervals_are_reproducible_from_the_seed():
    a = score_dataset(*_mixed_dataset(7, 5), bootstrap=1000, seed=3)
    b = score_dataset(*_mixed_dataset(7, 5), bootstrap=1000, seed=3)

    assert a.cis == b.cis


def test_the_seed_actually_drives_the_resampling():
    """Checked at n=120, not n=12: twelve filings admit only thirteen distinct accuracy values, so
    the percentile bounds land on the same points under most seeds. That coarseness is a property
    of the test set, not of the estimator."""
    a = score_dataset(*_mixed_dataset(100, 20), bootstrap=1000, seed=3)
    b = score_dataset(*_mixed_dataset(100, 20), bootstrap=1000, seed=4)

    assert a.cis["numeric_accuracy"] != b.cis["numeric_accuracy"]


def test_a_unanimous_result_has_a_degenerate_interval():
    report = score_dataset(*_mixed_dataset(5, 0), bootstrap=500)

    ci = report.cis["numeric_accuracy"]
    assert ci.low == pytest.approx(1.0) and ci.high == pytest.approx(1.0)


def test_bootstrap_can_be_disabled():
    report = score_dataset(*_mixed_dataset(3, 1), bootstrap=0)

    assert report.cis == {}
    assert report_to_dict(report)["bootstrap"] is None
    assert "CI" in format_report(report)  # column stays, cells read "--"


def test_metrics_no_resample_can_evaluate_are_omitted_not_zeroed():
    """Every filing satisfies Item 1A by cross-reference, so risk F1 is undefined -- reporting it
    as 0.0 with a [0, 0] interval would read as a measured failure."""
    target = make_target()
    examples = [make_example(target, by_reference=True) for _ in range(4)]
    report = score_dataset(examples, [target.model_dump_json()] * 4, bootstrap=500)

    assert "macro_category_f1" not in report.cis
    assert "numeric_accuracy" in report.cis


def test_intervals_survive_serialization_and_appear_in_the_report():
    report = score_dataset(*_mixed_dataset(9, 3), bootstrap=1000, model="ci-model")

    payload = report_to_dict(report)
    json.dumps(payload)
    assert payload["bootstrap"] == {"resamples": 10_000, "seed": 0, "unit": "filing"}
    interval = payload["confidence_intervals"]["numeric_accuracy"]
    assert interval["low"] < interval["high"]

    text = format_report(report)
    assert "95% CI" in text and "percentile bootstrap over filings" in text
