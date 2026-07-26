"""Report the real token-length distribution of the dataset, to set `max_seq_len` from data.

    python -m src.train.measure
    python -m src.train.measure --model-id meta-llama/Llama-3.2-1B-Instruct   # ungated stand-in

Character-count estimates are not good enough here: filings are number-dense, and digit sequences
tokenize far worse than prose. Getting this wrong truncates the target off the end of the longest
examples, which deletes their training signal without raising anything.
"""

from __future__ import annotations

import argparse
import statistics as stats
import sys

from src.train.data import PROCESSED_DIR, load_jsonl, render_prompt

SPLITS = ("train", "val", "test")


def measure(model_id: str, candidates: list[int]) -> None:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    lengths: list[tuple[str, int, int]] = []
    for split in SPLITS:
        path = PROCESSED_DIR / f"{split}.jsonl"
        if not path.exists():
            continue
        for row in load_jsonl(path):
            messages = row["messages"]
            assistant = next(m["content"] for m in messages if m["role"] == "assistant")
            prompt = len(tokenizer(render_prompt(tokenizer, messages))["input_ids"])
            completion = len(tokenizer(assistant)["input_ids"]) + 1  # + EOS
            lengths.append((row.get("ticker", "?"), prompt, completion))

    if not lengths:
        raise SystemExit(f"No splits found in {PROCESSED_DIR}. Run `python -m src.labels.build`.")

    totals = sorted(p + c for _, p, c in lengths)
    completions = [c for _, _, c in lengths]

    print(f"tokenizer: {model_id}   examples: {len(totals)}")
    print(
        f"total tokens   median {stats.median(totals):,.0f}   "
        f"p90 {totals[int(0.9 * len(totals))]:,}   p99 {totals[int(0.99 * len(totals))]:,}   "
        f"max {totals[-1]:,}"
    )
    print(
        f"target tokens  median {stats.median(completions):,.0f}   "
        f"max {max(completions):,}   "
        f"supervised share {sum(completions) / sum(totals):.1%}"
    )

    print("\nlongest examples:")
    for ticker, p, c in sorted(lengths, key=lambda t: t[1] + t[2], reverse=True)[:5]:
        print(f"  {ticker:6s} prompt {p:6,}  target {c:5,}  total {p + c:6,}")

    print("\nfit by candidate max_seq_len:")
    for limit in candidates:
        truncated = sum(1 for t in totals if t > limit)
        # A target that alone exceeds the window cannot be learned at any prompt length.
        impossible = sum(1 for c in completions if c >= limit)
        print(
            f"  {limit:6,}: {truncated:3d} of {len(totals)} examples truncated "
            f"({truncated / len(totals):.1%}){'  -- ' + str(impossible) + ' targets do not fit' if impossible else ''}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--candidates", type=int, nargs="+", default=[4096, 8192, 12288, 16384])
    args = parser.parse_args()
    sys.exit(measure(args.model_id, args.candidates) or 0)


if __name__ == "__main__":
    main()
