"""Assemble the labeled training dataset from cached filings.

    python -m src.labels.build                      # label everything in data/raw/
    python -m src.labels.build --limit 3            # smoke test
    python -m src.labels.build --no-teacher         # numerics only, no API spend
    python -m src.labels.build --dry-run            # excerpt + grounding stats, zero API calls

Each filing becomes one chat-format example: a system prompt describing the task, a user turn with
the bounded filing excerpt, and an assistant turn with the target JSON. Output is JSONL split by
company (src/labels/splits.py), written to data/processed/{train,val,test}.jsonl.

Resumable: filings already present in the output are skipped, so a partial run costs nothing to
resume. Teacher labels are cached per-filing in data/processed/_teacher_cache/ so that re-running
to change the *example* format never re-pays for labeling.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from collections import Counter
from pathlib import Path

from dotenv import load_dotenv

from src.data.schema import FilingSections, FinancialFacts
from src.labels.excerpt import build_excerpt, check_grounding
from src.labels.schema import ExtractionTarget, RiskFactor
from src.labels.splits import assign_split, split_counts
from src.labels.teacher import (
    AnthropicTeacher,
    Teacher,
    TeacherError,
    TeacherUnavailableError,
)

logger = logging.getLogger("label_builder")

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
TEACHER_CACHE_DIR = PROCESSED_DIR / "_teacher_cache"

SYSTEM_PROMPT = (
    "You extract structured data from SEC filings. Given an excerpt of a 10-K, return a single "
    "JSON object with the company's identifying details, its key financial figures, and the risk "
    "factors it discloses. Report financial figures in plain US dollars. Return only JSON."
)


def _load_filings() -> list[FilingSections]:
    filings = []
    for path in sorted(RAW_DIR.glob("*.json")):
        try:
            filings.append(FilingSections.model_validate_json(path.read_text()))
        except Exception as e:
            logger.error("Could not parse %s: %s", path.name, e)
    return filings


def _cache_path(accession_no: str, risk_text: str) -> Path:
    """Cache key includes a digest of the exact text that was labeled.

    Keying on accession alone would silently serve stale labels after any change to the excerpt
    budget -- the labels would describe risks no longer present in the student's input, which is
    the ungrounded-label failure this pipeline exists to avoid. A changed excerpt now misses the
    cache and re-labels instead.
    """
    digest = hashlib.sha256(risk_text.encode()).hexdigest()[:12]
    return TEACHER_CACHE_DIR / f"{accession_no.replace('/', '-')}_{digest}.json"


def _cached_labels(accession_no: str, risk_text: str) -> list[RiskFactor] | None:
    path = _cache_path(accession_no, risk_text)
    if not path.exists():
        return None
    try:
        raw = json.loads(path.read_text())
        return [RiskFactor.model_validate(r) for r in raw]
    except Exception as e:
        logger.warning("Ignoring corrupt teacher cache for %s: %s", accession_no, e)
        return None


def _write_cache(accession_no: str, risk_text: str, risks: list[RiskFactor]) -> None:
    TEACHER_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    _cache_path(accession_no, risk_text).write_text(
        json.dumps([r.model_dump() for r in risks], indent=2)
    )


def _risk_section_of(excerpt: str) -> str:
    """The Item 1A slice of the excerpt -- what the teacher labels.

    Labeling the excerpt rather than the full filing is deliberate: every risk factor in the target
    must be present in the model's input, or the example teaches hallucination.
    """
    marker = "## Item 1A. Risk Factors"
    idx = excerpt.find(marker)
    return excerpt[idx:] if idx != -1 else ""


# Item 1A may satisfy its disclosure requirement by pointing at another document -- Wells Fargo and
# US Bancorp both leave a ~200-character cross-reference to their Annual Report. The filing then
# contains no risk text to extract, so an empty risk list is the correct output, not a miss. These
# are flagged rather than dropped: the financial labels are still valid and grounded, and only
# risk-factor scoring needs to exclude them.
_REFERENCE_MARKERS = ("incorporated", "by reference", "can be found")
_REFERENCE_MAX_CHARS = 1_500


def is_incorporated_by_reference(risk_text: str | None) -> bool:
    if not risk_text:
        return False
    stripped = risk_text.strip()
    if len(stripped) > _REFERENCE_MAX_CHARS:
        return False
    lowered = stripped.lower()
    return any(marker in lowered for marker in _REFERENCE_MARKERS)


def grounded_financials(excerpt: str, facts: FinancialFacts) -> FinancialFacts:
    """Drop numeric labels the excerpt does not support.

    A figure that appears nowhere in the model's input cannot be derived from it, so keeping it as
    a target teaches the model to produce confident numbers from nothing -- the precise failure the
    grounding check exists to catch. `None` is a legitimate target: the schema already expects null
    for concepts a filer does not report, and Truist genuinely files no total-revenue tag.

    This also acts as the last line of defence against a mis-selected XBRL fact, which is how the
    convenience dict's 471M revenue for a bank would otherwise reach the training set.
    """
    grounded = check_grounding(excerpt, facts)
    kept = facts.model_dump()
    for field, ok in grounded.items():
        if not ok:
            kept[field] = None
    return FinancialFacts(**kept)


def build_example(sections: FilingSections, risk_factors: list[RiskFactor]) -> dict:
    """One chat-format training example."""
    excerpt = build_excerpt(sections)
    target = ExtractionTarget(
        company=sections.company_name,
        ticker=sections.ticker,
        fiscal_year=sections.fiscal_year,
        form_type=sections.form_type,
        financials=grounded_financials(excerpt, sections.financials),
        risk_factors=risk_factors,
    )
    return {
        "ticker": sections.ticker,
        "accession_no": sections.accession_no,
        # Consumed by src/eval/: exclude from risk-factor F1, keep for numeric scoring.
        "risk_factors_by_reference": is_incorporated_by_reference(sections.risk_factors),
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": excerpt},
            {"role": "assistant", "content": target.model_dump_json(indent=2)},
        ],
    }


def run(
    limit: int | None = None,
    use_teacher: bool = True,
    dry_run: bool = False,
    allow_partial: bool = False,
    teacher: Teacher | None = None,
) -> None:
    """`teacher` is injectable so tests can exercise failure paths without network access."""
    filings = _load_filings()
    if limit:
        filings = filings[:limit]
    if not filings:
        raise SystemExit(f"No filings found in {RAW_DIR}. Run `python -m src.data.pipeline` first.")

    if teacher is None and use_teacher and not dry_run:
        teacher = AnthropicTeacher()
    if dry_run or not use_teacher:
        teacher = None

    examples: dict[str, list[dict]] = {"train": [], "val": [], "test": []}
    grounding = Counter()
    grounded_total = Counter()
    risk_counts: list[int] = []
    category_counts = Counter()
    failures = 0
    aborted = False

    for i, sections in enumerate(filings, 1):
        logger.info("[%d/%d] %s (%s)", i, len(filings), sections.ticker, sections.accession_no)
        excerpt = build_excerpt(sections)

        for field, ok in check_grounding(excerpt, sections.financials).items():
            grounded_total[field] += 1
            grounding[field] += int(ok)

        risks: list[RiskFactor] = []
        if teacher is not None:
            risk_text = _risk_section_of(excerpt)
            risks = _cached_labels(sections.accession_no, risk_text) or []
            if risks:
                logger.info("  reusing cached teacher labels (%d risks)", len(risks))
            else:
                try:
                    risks = teacher.label_risk_factors(risk_text)
                    _write_cache(sections.accession_no, risk_text, risks)
                    # Ticker repeated on the result line: the progress line above prints when a
                    # filing *starts*, so a bare count here reads as belonging to the next ticker.
                    logger.info("  %s: labeled %d risk factors", sections.ticker, len(risks))
                except TeacherUnavailableError as e:
                    # Every remaining filing would fail the same way; stop rather than grind
                    # through the list producing identical errors.
                    logger.error("Teacher unavailable, aborting run: %s", e)
                    aborted = True
                    failures += 1
                    break
                except TeacherError as e:
                    logger.error("  teacher failed for %s: %s", sections.ticker, e)
                    failures += 1
                    continue

        risk_counts.append(len(risks))
        category_counts.update(r.category for r in risks)

        if not dry_run:
            split = assign_split(sections.ticker)
            examples[split].append(build_example(sections, risks))

    if aborted:
        logger.error(
            "Run aborted after %d of %d filings. Labels already obtained are cached.",
            len(risk_counts),
            len(filings),
        )

    # A failed run must never clobber a good dataset. Writing unconditionally means one billing
    # outage or a bad key silently replaces train.jsonl with an empty file -- which is exactly what
    # happened the first time this ran against a credit-exhausted account.
    if not dry_run and failures and not allow_partial:
        raise SystemExit(
            f"\n{failures} filing(s) failed to label; refusing to overwrite "
            f"{PROCESSED_DIR}/*.jsonl with a partial dataset.\n"
            "Fix the cause and re-run -- successful teacher labels are cached, so nothing is "
            "re-paid for. Pass --allow-partial to write anyway."
        )

    if not dry_run:
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        for split, rows in examples.items():
            out = PROCESSED_DIR / f"{split}.jsonl"
            with out.open("w") as f:
                for row in rows:
                    f.write(json.dumps(row) + "\n")
            logger.info("Wrote %d examples to %s", len(rows), out)

    _report(filings, grounding, grounded_total, risk_counts, category_counts, failures, teacher)


def _report(
    filings, grounding, grounded_total, risk_counts, category_counts, failures, teacher=None
) -> None:
    print("\n" + "=" * 62)
    print(f"Filings processed: {len(filings)}   teacher failures: {failures}")

    # Only an AnthropicTeacher tracks usage; a test double or a cache-only run has none.
    usage = getattr(teacher, "usage", None)
    if usage is not None and usage.calls:
        print(f"\nTeacher spend: {usage.summary(teacher.model)}")
        print("  Cached filings cost nothing, so this covers only the calls this run made.")

    print("\nCompany-level splits:")
    for split, n in split_counts([f.ticker for f in filings]).items():
        print(f"  {split:6s} {n:4d} companies")

    print("\nNumeric label grounding (does the figure appear in the excerpt?):")
    if not grounded_total:
        print("  no numeric labels present")
    for field, total in grounded_total.items():
        hit = grounding[field]
        print(f"  {field:28s} {hit:4d}/{total:<4d} ({hit / total:5.1%})")

    by_reference = [f.ticker for f in filings if is_incorporated_by_reference(f.risk_factors)]
    if by_reference:
        print(
            f"\nRisk factors incorporated by reference (no risk text in the filing): "
            f"{len(by_reference)} -- {', '.join(by_reference)}"
        )
        print("  flagged in the dataset; exclude from risk-factor scoring, keep for numerics.")

    if risk_counts:
        avg = sum(risk_counts) / len(risk_counts)
        print(f"\nRisk factors per filing: min {min(risk_counts)}  avg {avg:.1f}  max {max(risk_counts)}")
        print("\nCategory distribution:")
        for cat, n in category_counts.most_common():
            print(f"  {cat:24s} {n:4d}")
    print("=" * 62)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None, help="Only process the first N filings")
    parser.add_argument(
        "--no-teacher", action="store_true", help="Skip teacher labeling (numerics only, no cost)"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Report excerpt/grounding stats, write nothing"
    )
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Write the dataset even if some filings failed to label",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stdout,
    )
    load_dotenv()
    run(
        limit=args.limit,
        use_teacher=not args.no_teacher,
        dry_run=args.dry_run,
        allow_partial=args.allow_partial,
    )


if __name__ == "__main__":
    main()
