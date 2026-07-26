"""Run a model over a split and score it.

    python -m src.eval.run --backend anthropic                 # frontier zero-shot baseline
    python -m src.eval.run --backend hf --model-id meta-llama/Llama-3.1-8B-Instruct
    python -m src.eval.run --backend hf --model-id ... --adapter out/lora   # fine-tuned
    python -m src.eval.run --backend anthropic --limit 2                    # cheap smoke test

Writes a JSON report per model to data/eval/reports/ so runs can be compared later without
re-generating, and prints the same markdown table that goes into the README.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from dotenv import load_dotenv

from src.eval.metrics import format_report, load_split, report_to_dict, score_dataset
from src.eval.predict import (
    FRONTIER_MODEL,
    PROMPT_MODES,
    AnthropicModel,
    Model,
    estimate_cost,
    predict_dataset,
)

logger = logging.getLogger("eval")

ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = ROOT / "data" / "processed"
REPORT_DIR = ROOT / "data" / "eval" / "reports"


# A fine-tuned model must see the prompt it was trained on; anything else is out of distribution.
# A frontier model was never trained on the target shape and needs the schema. Explicit
# --prompt-mode overrides both.
DEFAULT_PROMPT_MODE = {"anthropic": "schema", "hf": "trained"}


def build_model(args) -> Model:
    system_prompt = PROMPT_MODES[args.prompt_mode]()

    if args.backend == "anthropic":
        return AnthropicModel(model=args.model_id or FRONTIER_MODEL, system_prompt=system_prompt)

    if not args.model_id:
        raise SystemExit("--model-id is required for the hf backend")
    from src.eval.predict import HFModel  # torch/transformers are an optional extra

    return HFModel(
        model_id=args.model_id,
        adapter_path=args.adapter,
        name=args.name,
        load_in_4bit=not args.no_4bit,
        system_prompt=system_prompt,
    )


def run(args) -> int:
    args.prompt_mode = args.prompt_mode or DEFAULT_PROMPT_MODE[args.backend]
    split_path = PROCESSED_DIR / f"{args.split}.jsonl"
    if not split_path.exists():
        raise SystemExit(f"{split_path} not found. Run `python -m src.labels.build` first.")

    examples = load_split(split_path)
    if args.limit:
        examples = examples[: args.limit]
    logger.info("Scoring %d examples from %s", len(examples), split_path.name)

    model = build_model(args)
    predictions, stats = predict_dataset(examples, model, use_cache=not args.no_cache)

    logger.info(
        "generated %d, cached %d, failed %d, truncated %d, refused %d",
        stats["generated"], stats["cached"], stats["failed"], stats["truncated"], stats["refused"],
    )

    # The prompt mode is part of the run's identity, not a footnote: the same model scores very
    # differently under the two, so a report that doesn't name the mode is unreadable later.
    name = args.name or f"{model.name} ({args.prompt_mode} prompt)"
    report = score_dataset(examples, predictions, model=name)
    print("\n" + format_report(report))

    payload = report_to_dict(report)
    payload["split"] = args.split
    payload["prompt_mode"] = args.prompt_mode
    payload["generation"] = stats
    if args.backend == "anthropic":
        cost = estimate_cost(stats)
        payload["estimated_cost_usd"] = cost
        if cost is not None:
            # Only the API backend has a token-denominated cost; a served model's is GPU-hours.
            print(f"\nEstimated generation cost: ${cost:.2f} "
                  f"({stats['input_tokens']:,} in / {stats['output_tokens']:,} out)")

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    slug = model.name.replace("/", "-")
    out = REPORT_DIR / f"{slug}_{args.prompt_mode}_{args.split}.json"
    out.write_text(json.dumps(payload, indent=2))
    print(f"\nWrote {out.relative_to(ROOT)}")

    # A run where generation itself failed is not a model result -- surface it in the exit code so
    # a scripted sweep doesn't record a partial run as a completed one.
    return 1 if stats["failed"] else 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--backend", default="anthropic", choices=["anthropic", "hf"])
    parser.add_argument("--model-id", default=None, help="Model name/path for the chosen backend")
    parser.add_argument("--adapter", default=None, help="LoRA adapter path (hf backend)")
    parser.add_argument("--name", default=None, help="Label for the report (defaults to model id)")
    parser.add_argument(
        "--prompt-mode",
        default=None,
        choices=sorted(PROMPT_MODES),
        help="'trained': the prompt the student was fine-tuned on. 'schema': that plus the JSON "
        "schema, for models that never saw the target shape. Defaults per backend: schema for "
        "anthropic, trained for hf (a fine-tuned model must see the prompt it was trained on).",
    )
    parser.add_argument("--limit", type=int, default=None, help="Only score the first N examples")
    parser.add_argument("--no-cache", action="store_true", help="Re-generate even if cached")
    parser.add_argument("--no-4bit", action="store_true", help="Load in bf16 instead of 4-bit")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stdout,
    )
    load_dotenv()
    raise SystemExit(run(args))


if __name__ == "__main__":
    main()
