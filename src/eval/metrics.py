"""Scoring for structured extraction. No model, no network -- text in, numbers out.

Three things are measured, in decreasing order of how much they can be trusted:

1. **Schema validity.** Does the output parse and validate as an `ExtractionTarget`? Reported both
   strictly (the raw string is JSON) and leniently (JSON survives after stripping markdown fences
   and prose). Base models routinely produce correct data wrapped in chatter, and collapsing those
   two cases into one number hides the single largest effect fine-tuning has.

2. **Numeric fields.** Ground truth is XBRL, so these are exact and provider-independent. Wrong
   answers are split by *how* they're wrong: a scale error (right digits, wrong magnitude) is a
   different defect from an invented figure, and the fix differs too.

3. **Risk factors.** Gold labels here came from a teacher model, so this measures agreement with
   the teacher, not correctness. Two metrics are reported: a category-multiset F1 that depends on
   no free parameters, and a per-risk F1 built on lexical overlap. The latter is a *lower bound* --
   a correct paraphrase sharing no vocabulary scores as a miss.

An unparseable prediction is scored as a total miss, not excluded. Dropping invalid outputs would
flatter a model that only manages valid JSON on the filings it finds easy.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Iterable, Optional

from pydantic import ValidationError

from src.data.schema import FinancialFacts
from src.labels.schema import ExtractionTarget, RiskFactor

# `shares_outstanding_diluted` is deliberately unscored: it is absent from most filings and is not
# covered by the grounding check, so it would inject noise rather than signal.
SCORED_FIELDS: tuple[str, ...] = ("revenue", "net_income", "total_assets", "eps_diluted")

# Treated as the same number: guards float round-tripping, not filer rounding.
EXACT_TOL = 1e-9
CLOSE_TOL = 0.01  # 1% -- covers a filing that writes "22.2" for an XBRL 22.18

# Filings report "in thousands" / "in millions"; a model that copies the printed digits without
# rescaling lands on one of these ratios exactly.
_SCALE_RATIOS = (1e3, 1e6, 1e9, 1e-3, 1e-6, 1e-9)


# --------------------------------------------------------------------------------------------
# Parsing model output
# --------------------------------------------------------------------------------------------

_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def extract_json(raw: str) -> Optional[str]:
    """Best-effort recovery of a JSON object from free-form model output.

    Handles the two things models actually do: wrap the object in a ```json fence, or bracket it
    with prose. Returns None when no balanced object is present.
    """
    if not raw:
        return None

    fenced = _FENCE.search(raw)
    candidate = fenced.group(1) if fenced else raw

    start = candidate.find("{")
    if start == -1:
        return None

    # Walk to the matching brace, ignoring braces inside string literals.
    depth = 0
    in_string = False
    escaped = False
    for i, ch in enumerate(candidate[start:], start=start):
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return candidate[start : i + 1]
    return None


@dataclass
class ParsedPrediction:
    """The outcome of turning one raw model response into a target object."""

    raw: str
    target: Optional[ExtractionTarget] = None
    strict_valid: bool = False  # the raw string itself is a valid target
    lenient_valid: bool = False  # a valid target was recoverable from it
    error: Optional[str] = None

    @property
    def usable(self) -> bool:
        return self.target is not None


def parse_prediction(raw: str) -> ParsedPrediction:
    def _validate(text: str) -> tuple[Optional[ExtractionTarget], Optional[str]]:
        try:
            return ExtractionTarget.model_validate_json(text), None
        except ValidationError as e:
            return None, f"schema: {e.error_count()} validation error(s)"
        except ValueError as e:
            return None, f"json: {e}"

    strict_target, strict_err = _validate(raw.strip()) if raw and raw.strip() else (None, "empty")
    if strict_target is not None:
        return ParsedPrediction(raw=raw, target=strict_target, strict_valid=True, lenient_valid=True)

    recovered = extract_json(raw)
    if recovered is None:
        return ParsedPrediction(raw=raw, error=strict_err or "no JSON object found")

    target, err = _validate(recovered)
    if target is None:
        return ParsedPrediction(raw=raw, error=err)
    return ParsedPrediction(raw=raw, target=target, lenient_valid=True)


# --------------------------------------------------------------------------------------------
# Numeric fields
# --------------------------------------------------------------------------------------------

# Per-field verdicts. Ordered loosely worst-to-best for reporting.
NOT_SCORED = "not_scored"  # gold is None -- the filer never reported the concept
SPURIOUS = "spurious"  # gold is None but the model produced a figure anyway
MISSING = "missing"  # gold present, model returned null
SCALE_ERROR = "scale_error"  # right digits, wrong magnitude
WRONG = "wrong"
CLOSE = "close"  # within CLOSE_TOL, e.g. filer-rounded
EXACT = "exact"

VERDICTS: tuple[str, ...] = (EXACT, CLOSE, SCALE_ERROR, WRONG, MISSING, SPURIOUS, NOT_SCORED)


def _relative_error(gold: float, pred: float) -> float:
    if gold == 0:
        return abs(pred)
    return abs(pred - gold) / abs(gold)


def is_scale_error(gold: float, pred: float) -> bool:
    """True when pred equals gold off by a clean power-of-ten reporting scale."""
    if pred == 0 or gold == 0:
        return False
    if (pred < 0) != (gold < 0):
        return False
    ratio = abs(gold) / abs(pred)
    return any(math.isclose(ratio, r, rel_tol=1e-6) for r in _SCALE_RATIOS)


def score_field(gold: Optional[float], pred: Optional[float]) -> str:
    if gold is None:
        return SPURIOUS if pred is not None else NOT_SCORED
    if pred is None:
        return MISSING
    rel = _relative_error(gold, pred)
    if rel <= EXACT_TOL:
        return EXACT
    if rel <= CLOSE_TOL:
        return CLOSE
    if is_scale_error(gold, pred):
        return SCALE_ERROR
    return WRONG


def score_numerics(gold: FinancialFacts, pred: Optional[FinancialFacts]) -> dict[str, str]:
    """Per-field verdicts. A `None` prediction (invalid output) scores every field as a miss."""
    return {
        f: score_field(getattr(gold, f), getattr(pred, f) if pred is not None else None)
        for f in SCORED_FIELDS
    }


# --------------------------------------------------------------------------------------------
# Risk factors
# --------------------------------------------------------------------------------------------

# Filing boilerplate that appears in nearly every risk title and would inflate overlap between
# unrelated risks ("Risks related to our business" vs "Risks related to our operations").
_STOPWORDS = frozenset(
    """a an and any are as at be been but by can could for from has have if in into is it its
    may might must not of on or our ours that the their them these this those to us we were which
    will with would risk risks related relating result results affect affected adversely material
    materially business company operations operating financial condition""".split()
)

RISK_MATCH_THRESHOLD = 0.30
_TITLE_WEIGHT = 0.7  # titles carry the identity of a risk; summaries only corroborate


def _tokens(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", text.lower()) if len(w) > 2 and w not in _STOPWORDS}


def _dice(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return 2 * len(a & b) / (len(a) + len(b))


def risk_similarity(a: RiskFactor, b: RiskFactor) -> float:
    """Lexical overlap in [0, 1], title-weighted. A deliberately simple, deterministic proxy --
    see the module docstring on why the resulting F1 is a lower bound."""
    title = _dice(_tokens(a.title), _tokens(b.title))
    summary = _dice(_tokens(a.summary), _tokens(b.summary))
    return _TITLE_WEIGHT * title + (1 - _TITLE_WEIGHT) * summary


def match_risks(
    gold: list[RiskFactor], pred: list[RiskFactor], threshold: float = RISK_MATCH_THRESHOLD
) -> list[tuple[int, int, float]]:
    """Greedy highest-similarity-first one-to-one matching. Returns (gold_idx, pred_idx, score).

    Greedy rather than optimal (Hungarian): with a similarity threshold this high the assignments
    rarely conflict, and it keeps the metric dependency-free and easy to reason about.
    """
    pairs = [
        (risk_similarity(g, p), gi, pi)
        for gi, g in enumerate(gold)
        for pi, p in enumerate(pred)
    ]
    pairs.sort(key=lambda t: (-t[0], t[1], t[2]))

    used_gold: set[int] = set()
    used_pred: set[int] = set()
    matches = []
    for score, gi, pi in pairs:
        if score < threshold:
            break
        if gi in used_gold or pi in used_pred:
            continue
        used_gold.add(gi)
        used_pred.add(pi)
        matches.append((gi, pi, score))
    return matches


@dataclass
class RiskScore:
    n_gold: int
    n_pred: int
    # Category-multiset agreement: parameter-free, insensitive to wording.
    category_tp: int = 0
    # Per-risk matching at RISK_MATCH_THRESHOLD.
    matched: int = 0
    matched_same_category: int = 0

    @staticmethod
    def _prf(tp: int, n_pred: int, n_gold: int) -> tuple[float, float, float]:
        precision = tp / n_pred if n_pred else 0.0
        recall = tp / n_gold if n_gold else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        return precision, recall, f1

    @property
    def category_f1(self) -> float:
        return self._prf(self.category_tp, self.n_pred, self.n_gold)[2]

    @property
    def match_f1(self) -> float:
        return self._prf(self.matched, self.n_pred, self.n_gold)[2]

    @property
    def category_accuracy(self) -> Optional[float]:
        """Among matched risks, the share the model also categorized the same way. None when
        nothing matched -- 0.0 would read as "categorized everything wrong"."""
        return self.matched_same_category / self.matched if self.matched else None


def score_risks(gold: list[RiskFactor], pred: list[RiskFactor]) -> RiskScore:
    gold_cats = Counter(r.category for r in gold)
    pred_cats = Counter(r.category for r in pred)
    category_tp = sum((gold_cats & pred_cats).values())

    matches = match_risks(gold, pred)
    same_cat = sum(1 for gi, pi, _ in matches if gold[gi].category == pred[pi].category)

    return RiskScore(
        n_gold=len(gold),
        n_pred=len(pred),
        category_tp=category_tp,
        matched=len(matches),
        matched_same_category=same_cat,
    )


# --------------------------------------------------------------------------------------------
# Per-example and corpus scoring
# --------------------------------------------------------------------------------------------


@dataclass
class ExampleScore:
    ticker: str
    accession_no: str
    strict_valid: bool
    lenient_valid: bool
    numerics: dict[str, str]
    risk: Optional[RiskScore]  # None when risk scoring is not applicable to this filing
    parse_error: Optional[str] = None


def gold_target(example: dict) -> ExtractionTarget:
    """The assistant turn of a dataset row, as a target object."""
    return ExtractionTarget.model_validate_json(example["messages"][-1]["content"])


def score_example(example: dict, raw_prediction: str) -> ExampleScore:
    gold = gold_target(example)
    parsed = parse_prediction(raw_prediction)
    pred = parsed.target

    # Item 1A satisfied by cross-reference: the filing carries no risk text, so there is nothing to
    # extract and nothing to score. Numerics still count -- see src/labels/build.py.
    risk = None
    if not example.get("risk_factors_by_reference", False):
        risk = score_risks(gold.risk_factors, pred.risk_factors if pred else [])

    return ExampleScore(
        ticker=example.get("ticker", "?"),
        accession_no=example.get("accession_no", "?"),
        strict_valid=parsed.strict_valid,
        lenient_valid=parsed.lenient_valid,
        numerics=score_numerics(gold.financials, pred.financials if pred else None),
        risk=risk,
        parse_error=parsed.error,
    )


@dataclass
class EvalReport:
    model: str
    n: int = 0
    strict_valid: int = 0
    lenient_valid: int = 0
    # field -> verdict -> count
    numerics: dict[str, Counter] = field(default_factory=dict)
    n_risk_scored: int = 0
    risk_micro: RiskScore = field(default_factory=lambda: RiskScore(0, 0))
    macro_category_f1: float = 0.0
    macro_match_f1: float = 0.0
    macro_category_accuracy: Optional[float] = None
    examples: list[ExampleScore] = field(default_factory=list)

    def numeric_rate(self, field_name: str, verdicts: Iterable[str] = (EXACT, CLOSE)) -> Optional[float]:
        """Share of *scorable* instances (gold present) landing in `verdicts`."""
        tally = self.numerics.get(field_name, Counter())
        scorable = sum(v for k, v in tally.items() if k != NOT_SCORED and k != SPURIOUS)
        if not scorable:
            return None
        return sum(tally[v] for v in verdicts) / scorable

    @property
    def numeric_accuracy(self) -> Optional[float]:
        """Headline numeric number: exact-or-close across all scorable fields."""
        hits = scorable = 0
        for tally in self.numerics.values():
            scorable += sum(v for k, v in tally.items() if k not in (NOT_SCORED, SPURIOUS))
            hits += tally[EXACT] + tally[CLOSE]
        return hits / scorable if scorable else None


def score_dataset(
    examples: list[dict], predictions: list[str], model: str = "unknown"
) -> EvalReport:
    if len(examples) != len(predictions):
        raise ValueError(f"{len(examples)} examples but {len(predictions)} predictions")

    report = EvalReport(model=model, n=len(examples))
    report.numerics = {f: Counter() for f in SCORED_FIELDS}

    cat_f1s: list[float] = []
    match_f1s: list[float] = []
    cat_accs: list[float] = []

    for example, raw in zip(examples, predictions):
        score = score_example(example, raw)
        report.examples.append(score)
        report.strict_valid += int(score.strict_valid)
        report.lenient_valid += int(score.lenient_valid)
        for f, verdict in score.numerics.items():
            report.numerics[f][verdict] += 1

        if score.risk is not None:
            report.n_risk_scored += 1
            report.risk_micro.n_gold += score.risk.n_gold
            report.risk_micro.n_pred += score.risk.n_pred
            report.risk_micro.category_tp += score.risk.category_tp
            report.risk_micro.matched += score.risk.matched
            report.risk_micro.matched_same_category += score.risk.matched_same_category
            cat_f1s.append(score.risk.category_f1)
            match_f1s.append(score.risk.match_f1)
            if score.risk.category_accuracy is not None:
                cat_accs.append(score.risk.category_accuracy)

    # Macro (per-filing mean) is the headline: one verbose filer discloses 46 risks where another
    # discloses 5, and micro-averaging would let that single filing set the corpus number.
    report.macro_category_f1 = sum(cat_f1s) / len(cat_f1s) if cat_f1s else 0.0
    report.macro_match_f1 = sum(match_f1s) / len(match_f1s) if match_f1s else 0.0
    report.macro_category_accuracy = sum(cat_accs) / len(cat_accs) if cat_accs else None
    return report


# --------------------------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------------------------


def _pct(value: Optional[float]) -> str:
    return "--" if value is None else f"{value:.1%}"


def format_report(report: EvalReport) -> str:
    """Markdown, so the same text serves the console and the README results table."""
    lines = [
        f"### {report.model}  (n={report.n})",
        "",
        "| Metric | Value |",
        "| --- | --- |",
        f"| Schema valid (strict) | {_pct(report.strict_valid / report.n if report.n else None)} |",
        f"| Schema valid (after fence/prose stripping) | {_pct(report.lenient_valid / report.n if report.n else None)} |",
        f"| Numeric accuracy (all fields) | {_pct(report.numeric_accuracy)} |",
        f"| Risk category F1 (macro) | {_pct(report.macro_category_f1)} |",
        f"| Risk match F1 (macro, lower bound) | {_pct(report.macro_match_f1)} |",
        f"| Category accuracy among matched | {_pct(report.macro_category_accuracy)} |",
        "",
        "| Numeric field | exact | close | scale err | wrong | missing | spurious | n/a |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for f in SCORED_FIELDS:
        t = report.numerics.get(f, Counter())
        lines.append(
            f"| {f} | {t[EXACT]} | {t[CLOSE]} | {t[SCALE_ERROR]} | {t[WRONG]} | "
            f"{t[MISSING]} | {t[SPURIOUS]} | {t[NOT_SCORED]} |"
        )

    skipped = report.n - report.n_risk_scored
    if skipped:
        clause = (
            "1 filing satisfies Item 1A by cross-reference and contains no risk text"
            if skipped == 1
            else f"{skipped} filings satisfy Item 1A by cross-reference and contain no risk text"
        )
        lines += [
            "",
            f"_Risk scoring covers {report.n_risk_scored} of {report.n} filings; {clause}._",
        ]
    return "\n".join(lines)


def report_to_dict(report: EvalReport) -> dict:
    """Serializable summary, for data/eval/*.json and cross-run comparison."""
    return {
        "model": report.model,
        "n": report.n,
        "strict_valid_rate": report.strict_valid / report.n if report.n else None,
        "lenient_valid_rate": report.lenient_valid / report.n if report.n else None,
        "numeric_accuracy": report.numeric_accuracy,
        "numeric_by_field": {
            f: {**{v: report.numerics[f][v] for v in VERDICTS}, "accuracy": report.numeric_rate(f)}
            for f in SCORED_FIELDS
        },
        "risk": {
            "n_scored": report.n_risk_scored,
            "macro_category_f1": report.macro_category_f1,
            "macro_match_f1": report.macro_match_f1,
            "macro_category_accuracy": report.macro_category_accuracy,
            "micro_category_f1": report.risk_micro.category_f1,
            "micro_match_f1": report.risk_micro.match_f1,
            "n_gold": report.risk_micro.n_gold,
            "n_pred": report.risk_micro.n_pred,
        },
        "per_filing": [
            {
                "ticker": e.ticker,
                "strict_valid": e.strict_valid,
                "lenient_valid": e.lenient_valid,
                "numerics": e.numerics,
                "risk_category_f1": e.risk.category_f1 if e.risk else None,
                "risk_match_f1": e.risk.match_f1 if e.risk else None,
                "parse_error": e.parse_error,
            }
            for e in report.examples
        ],
    }


def load_split(path) -> list[dict]:
    from pathlib import Path

    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
