"""Ingestion entrypoint: download + parse the latest 10-K for every company in
src/data/companies.py, caching one JSON file per filing under data/raw/.

Usage:
    python -m src.data.pipeline
    python -m src.data.pipeline --limit 10          # smoke test on a handful of tickers
    python -m src.data.pipeline --tickers AAPL MSFT  # specific tickers only

Resumable: existing output files are skipped, so re-running after a partial failure only fetches
what's missing. Failures are logged to data/raw/_failures.log with the ticker and reason rather
than silently dropped, per the project plan.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

from src.data.companies import TICKERS
from src.data.download import FilingFetchError, get_latest_10ks, parse_filing

logger = logging.getLogger("edgar_pipeline")

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
FAILURES_LOG = RAW_DIR / "_failures.log"


def _configure_identity() -> None:
    load_dotenv()
    identity = os.environ.get("EDGAR_IDENTITY")
    if not identity:
        raise SystemExit(
            "EDGAR_IDENTITY is not set. Copy .env.example to .env and fill it in "
            '(e.g. "Your Name you@example.com") -- SEC blocks requests without it.'
        )
    import edgar

    edgar.set_identity(identity)


def _output_path(ticker: str, accession_no: str) -> Path:
    safe_accession = accession_no.replace("/", "-")
    return RAW_DIR / f"{ticker}_{safe_accession}.json"


def _log_failure(ticker: str, reason: str) -> None:
    FAILURES_LOG.parent.mkdir(parents=True, exist_ok=True)
    with FAILURES_LOG.open("a") as f:
        f.write(f"{ticker}\t{reason}\n")


def ingest_ticker(ticker: str, n_filings: int = 1) -> int:
    """Returns the number of filings newly written for this ticker."""
    try:
        filings = get_latest_10ks(ticker, n=n_filings)
    except FilingFetchError as e:
        logger.error("Skipping %s: could not list filings (%s)", ticker, e)
        _log_failure(ticker, f"list_filings_failed: {e}")
        return 0

    written = 0
    for filing in filings:
        out_path = _output_path(ticker, filing.accession_no)
        if out_path.exists():
            logger.info("Skipping %s (%s) -- already cached", ticker, filing.accession_no)
            continue
        try:
            sections = parse_filing(ticker, filing)
        except FilingFetchError as e:
            logger.error("Skipping %s %s: %s", ticker, filing.accession_no, e)
            _log_failure(ticker, f"{filing.accession_no}: parse_failed: {e}")
            continue

        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(sections.model_dump_json(indent=2))
        written += 1
        logger.info(
            "Wrote %s (%s) -- warnings: %s",
            out_path.name,
            filing.accession_no,
            sections.parse_warnings or "none",
        )
    return written


def run(tickers: list[str], n_filings: int = 1, sleep_seconds: float = 0.3) -> None:
    _configure_identity()
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    total_written = 0
    for i, ticker in enumerate(tickers, 1):
        logger.info("[%d/%d] %s", i, len(tickers), ticker)
        total_written += ingest_ticker(ticker, n_filings=n_filings)
        time.sleep(sleep_seconds)

    logger.info("Done. %d new filings written to %s", total_written, RAW_DIR)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tickers", nargs="+", default=None, help="Specific tickers to ingest")
    parser.add_argument("--limit", type=int, default=None, help="Only process the first N tickers")
    parser.add_argument("--n-filings", type=int, default=1, help="10-Ks per company (newest first)")
    parser.add_argument("--sleep", type=float, default=0.3, help="Seconds to sleep between companies")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stdout,
    )

    tickers = args.tickers if args.tickers else list(TICKERS)
    if args.limit:
        tickers = tickers[: args.limit]

    run(tickers, n_filings=args.n_filings, sleep_seconds=args.sleep)


if __name__ == "__main__":
    main()
