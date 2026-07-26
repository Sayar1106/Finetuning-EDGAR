"""Company-level train/val/test splitting.

The anti-leakage rule from the project plan: split by **company**, never by filing. A company that
files a 10-K every year uses near-identical boilerplate across years -- splitting by filing would
put FY2024 Apple in train and FY2025 Apple in test, and the resulting score would measure
memorization of Apple's phrasing rather than generalization to unseen companies.

Assignment is a deterministic hash of the ticker, so it is stable across runs and across changes to
the ticker list -- adding companies never reshuffles the existing ones into different splits.
"""

from __future__ import annotations

import hashlib

Split = str  # "train" | "val" | "test"

DEFAULT_RATIOS = {"train": 0.8, "val": 0.1, "test": 0.1}


def _ticker_bucket(ticker: str) -> float:
    """Stable, uniformly distributed value in [0, 1) derived from the ticker."""
    digest = hashlib.sha256(ticker.upper().encode()).hexdigest()
    return int(digest[:16], 16) / float(1 << 64)


def assign_split(ticker: str, ratios: dict[str, float] | None = None) -> Split:
    """Deterministically assign one company to a split."""
    ratios = ratios or DEFAULT_RATIOS
    bucket = _ticker_bucket(ticker)

    cumulative = 0.0
    for name in ("train", "val", "test"):
        cumulative += ratios[name]
        if bucket < cumulative:
            return name
    return "test"


def split_counts(tickers: list[str], ratios: dict[str, float] | None = None) -> dict[str, int]:
    """Companies per split, for reporting dataset composition."""
    counts = {"train": 0, "val": 0, "test": 0}
    for ticker in tickers:
        counts[assign_split(ticker, ratios)] += 1
    return counts
