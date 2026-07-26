"""Re-extract XBRL facts for already-ingested filings, in place.

    python -m src.data.refresh_facts --dry-run    # report what would change
    python -m src.data.refresh_facts              # rewrite data/raw/*.json

Only the `financials` block and its warnings are rewritten. Section text is left byte-identical on
purpose: the excerpt is what the teacher labeled, and the teacher cache is keyed on a digest of
that text (src/labels/build.py). Re-running the full pipeline risks a whitespace-level change in a
parsed section, which would miss the cache and re-pay for labeling 182 filings.

Use this after changing fact selection in src/data/xbrl_facts.py. For a fresh corpus, use
`python -m src.data.pipeline` instead.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from dotenv import load_dotenv

from src.data.download import FilingFetchError, get_latest_10ks
from src.data.schema import FilingSections
from src.data.xbrl_facts import extract_financial_facts

logger = logging.getLogger("refresh_facts")

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"

FIELDS = ("revenue", "net_income", "total_assets", "eps_diluted")


def _changed(before, after) -> list[str]:
    diffs = []
    for field in FIELDS:
        old, new = getattr(before, field), getattr(after, field)
        if old != new:
            diffs.append(f"{field}: {old} -> {new}")
    return diffs


def run(dry_run: bool = False, limit: int | None = None) -> int:
    paths = sorted(RAW_DIR.glob("*.json"))
    if limit:
        paths = paths[:limit]
    if not paths:
        raise SystemExit(f"No filings in {RAW_DIR}. Run `python -m src.data.pipeline` first.")

    changed = failed = 0
    for i, path in enumerate(paths, 1):
        sections = FilingSections.model_validate_json(path.read_text())
        try:
            filings = get_latest_10ks(sections.ticker, 1)
            filing = next((f for f in filings if f.accession_no == sections.accession_no), None)
            if filing is None:
                # A newer 10-K has been filed since ingestion; refreshing from it would silently
                # relabel this row against a different document than the cached excerpt.
                logger.warning(
                    "[%d/%d] %s: accession %s no longer latest, skipping",
                    i, len(paths), sections.ticker, sections.accession_no,
                )
                failed += 1
                continue
            facts, warnings = extract_financial_facts(filing)
        except FilingFetchError as e:
            logger.error("[%d/%d] %s: %s", i, len(paths), sections.ticker, e)
            failed += 1
            continue

        diffs = _changed(sections.financials, facts)
        if diffs:
            changed += 1
            logger.info("[%d/%d] %s: %s", i, len(paths), sections.ticker, "; ".join(diffs))
        else:
            logger.debug("[%d/%d] %s: unchanged", i, len(paths), sections.ticker)

        if not dry_run:
            sections.financials = facts
            kept = [w for w in sections.parse_warnings if not _is_facts_warning(w)]
            sections.parse_warnings = kept + warnings
            path.write_text(sections.model_dump_json(indent=2))

    verb = "would change" if dry_run else "changed"
    print(f"\n{changed} of {len(paths)} filings {verb}; {failed} could not be refreshed.")
    return 1 if failed else 0


_FACTS_WARNING_MARKERS = ("financial_metrics_failed", "financial_facts_failed", "xbrl_unavailable",
                          "_from_metrics_fallback", "eps_diluted_derived_from")


def _is_facts_warning(warning: str) -> bool:
    """Facts warnings are replaced wholesale on refresh; parse warnings for the text sections are
    about the cached document and must survive."""
    return any(marker in warning for marker in _FACTS_WARNING_MARKERS)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Report changes, write nothing")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stdout,
    )
    load_dotenv()
    raise SystemExit(run(dry_run=args.dry_run, limit=args.limit))


if __name__ == "__main__":
    main()
