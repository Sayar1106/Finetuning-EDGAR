"""Generate extractions from a model, so src/eval/metrics.py can score them.

Every backend answers the same question -- given one filing excerpt, return the raw text the model
produced -- and returns that text *unmodified*. Repair and validation belong to the scorer; a
backend that quietly fixed its own JSON would erase the schema-validity metric.

Two fairness rules govern the baselines, both of which affect published numbers:

1. **The baseline gets the schema.** Two prompt modes exist, and the distinction decides whether
   the frontier comparison means anything:

   - `trained` -- the exact system prompt the student was fine-tuned on: prose, no field names.
   - `schema` -- that prompt plus the JSON schema of the extraction target.

   The student does not need the schema in its prompt because 152 training examples put it in the
   weights. Giving a frontier model the same information in-context is the equivalent affordance,
   not a handicap removed. Measured on `trained` alone, Sonnet invents its own field names
   (`total_revenues`, `diluted_eps`) and scores zero -- a real finding about what fine-tuning
   teaches, but a strawman as a competitive baseline. Both are run; both are reported.

2. **No constrained decoding, for anyone.** The Anthropic backend deliberately does *not* use
   structured outputs, even though it could. Constrained decoding would hand the baseline a 100%
   schema-validity score by construction and make the headline comparison meaningless. It is
   equally available to a served Llama (vLLM guided decoding), so the honest comparison is
   unconstrained-vs-unconstrained, with the caveat stated in the README.

Predictions are cached per (model, filing, prompt) so re-scoring never re-pays for generation.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Protocol

from src.labels.build import SYSTEM_PROMPT
from src.labels.schema import ExtractionTarget

logger = logging.getLogger(__name__)


def schema_prompt() -> str:
    """`SYSTEM_PROMPT` plus the target's JSON schema, for models that were never trained on it."""
    schema = json.dumps(ExtractionTarget.model_json_schema(), indent=2)
    return (
        f"{SYSTEM_PROMPT}\n\n"
        "The JSON object must conform to this schema. Use exactly these field names, and emit "
        "null for any figure the excerpt does not state.\n\n"
        f"{schema}"
    )


PROMPT_MODES = {"trained": lambda: SYSTEM_PROMPT, "schema": schema_prompt}

ROOT = Path(__file__).resolve().parents[2]
EVAL_DIR = ROOT / "data" / "eval"
PREDICTION_CACHE = EVAL_DIR / "predictions"

# Long risk lists are real: one test filing discloses 46 risks. Truncation would be scored as an
# invalid prediction, so the budget is set well above the largest observed target.
MAX_OUTPUT_TOKENS = 8_000

# The frontier baseline. Deliberately NOT Haiku: Haiku 4.5 generated the risk-factor gold labels,
# so scoring Haiku here would measure self-agreement and report a meaningless near-perfect F1.
FRONTIER_MODEL = "claude-sonnet-5"

# Sonnet 5 thinks by default, and thinking shares the max_tokens budget with the answer. The
# baseline is deliberately left in its default configuration -- handicapping it would make
# "matched a frontier model" a claim about a weakened opponent -- so the budget is doubled to
# leave room for both. Cost of the choice is recorded per run (see `estimate_cost`).
FRONTIER_MAX_TOKENS = 16_000

# $ per million tokens, as of 2026-07. Sonnet 5 introductory rate ($2/$10) runs through
# 2026-08-31; the standard rate is $3/$15. Used only to report the cost side of the comparison.
FRONTIER_PRICE_PER_MTOK = {"input": 2.0, "output": 10.0}


class PredictionError(Exception):
    """Generation failed for one filing. The caller records it and moves on."""


class Model(Protocol):
    """Anything that turns a filing excerpt into raw extraction text."""

    name: str
    system_prompt: str  # part of the cache key: a different prompt is a different experiment

    def predict(self, excerpt: str) -> "Prediction": ...


@dataclass
class Prediction:
    raw: str
    stop_reason: Optional[str] = None
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None

    @property
    def truncated(self) -> bool:
        return self.stop_reason in ("max_tokens", "length")

    @property
    def refused(self) -> bool:
        """A safety-classifier decline. Returns HTTP 200 with empty content, so it looks like a
        blank answer unless checked -- and it is a run-quality signal, not a model-quality one."""
        return self.stop_reason == "refusal"

    def to_dict(self) -> dict:
        return {
            "raw": self.raw,
            "stop_reason": self.stop_reason,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "Prediction":
        return cls(
            raw=payload["raw"],
            stop_reason=payload.get("stop_reason"),
            input_tokens=payload.get("input_tokens"),
            output_tokens=payload.get("output_tokens"),
        )


# --------------------------------------------------------------------------------------------
# Backends
# --------------------------------------------------------------------------------------------


class AnthropicModel:
    """Frontier zero-shot baseline over the Messages API."""

    def __init__(
        self,
        model: str = FRONTIER_MODEL,
        max_retries: int = 3,
        system_prompt: str | None = None,
    ):
        import anthropic

        try:
            self._client = anthropic.Anthropic()
        except Exception as e:
            raise SystemExit(
                f"Could not construct an Anthropic client ({e}). Set ANTHROPIC_API_KEY in .env, "
                "or run `ant auth login` to store a profile."
            ) from e
        self.name = model
        self.system_prompt = system_prompt or schema_prompt()
        self._max_retries = max_retries

    def predict(self, excerpt: str) -> Prediction:
        import anthropic

        last_exc: Exception | None = None
        for attempt in range(self._max_retries):
            try:
                # No `temperature`: current models reject a non-default value outright. No
                # `thinking`: the model's own default is the honest baseline configuration.
                response = self._client.messages.create(
                    model=self.name,
                    max_tokens=FRONTIER_MAX_TOKENS,
                    system=self.system_prompt,
                    messages=[{"role": "user", "content": excerpt}],
                )
                text = "".join(b.text for b in response.content if b.type == "text")
                return Prediction(
                    raw=text,
                    stop_reason=response.stop_reason,
                    input_tokens=response.usage.input_tokens,
                    output_tokens=response.usage.output_tokens,
                )
            except (
                anthropic.BadRequestError,
                anthropic.AuthenticationError,
                anthropic.PermissionDeniedError,
                anthropic.NotFoundError,
            ) as e:
                # Exhausted credit, bad key, unknown model: every remaining filing fails the same
                # way, so surface it now instead of retrying 12 times.
                raise SystemExit(f"Anthropic API unavailable: {e}") from e
            except Exception as e:
                last_exc = e
                delay = 2.0 * (2**attempt)
                logger.warning(
                    "generation failed (attempt %d/%d): %s -- retrying in %.1fs",
                    attempt + 1,
                    self._max_retries,
                    e,
                    delay,
                )
                time.sleep(delay)

        raise PredictionError(f"generation failed after {self._max_retries} attempts: {last_exc}")


class HFModel:
    """Local transformers backend for base and fine-tuned Llama.

    Greedy decoding: the metrics compare extraction accuracy, and sampling would make run-to-run
    differences indistinguishable from the effect of fine-tuning.
    """

    def __init__(
        self,
        model_id: str,
        adapter_path: str | None = None,
        name: str | None = None,
        load_in_4bit: bool = True,
        max_new_tokens: int = MAX_OUTPUT_TOKENS,
        system_prompt: str | None = None,
    ):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.name = name or (adapter_path or model_id)
        # The fine-tuned model is trained on SYSTEM_PROMPT, so that is the default here. The
        # untrained base model is a fair candidate for either mode -- see `--prompt-mode`.
        self.system_prompt = system_prompt or SYSTEM_PROMPT
        self._max_new_tokens = max_new_tokens

        kwargs: dict = {"torch_dtype": torch.bfloat16, "device_map": "auto"}
        if load_in_4bit:
            from transformers import BitsAndBytesConfig

            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
            )

        self._tokenizer = AutoTokenizer.from_pretrained(model_id)
        self._model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)

        if adapter_path:
            from peft import PeftModel

            self._model = PeftModel.from_pretrained(self._model, adapter_path)
        self._model.eval()

    def predict(self, excerpt: str) -> Prediction:
        import torch

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": excerpt},
        ]
        prompt = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)

        with torch.no_grad():
            output = self._model.generate(
                **inputs,
                max_new_tokens=self._max_new_tokens,
                do_sample=False,
                pad_token_id=self._tokenizer.eos_token_id,
            )

        generated = output[0][inputs["input_ids"].shape[1] :]
        text = self._tokenizer.decode(generated, skip_special_tokens=True)
        # `generate` stops at max_new_tokens without saying so; infer it from the length.
        stop_reason = "max_tokens" if len(generated) >= self._max_new_tokens else "end_turn"
        return Prediction(
            raw=text,
            stop_reason=stop_reason,
            input_tokens=int(inputs["input_ids"].shape[1]),
            output_tokens=int(len(generated)),
        )


# --------------------------------------------------------------------------------------------
# Caching + batch generation
# --------------------------------------------------------------------------------------------


def _slug(name: str) -> str:
    return name.replace("/", "-").replace(":", "-")


def cache_path(model_name: str, accession_no: str, excerpt: str, system_prompt: str) -> Path:
    """Keyed on the full prompt, not just the filing.

    The system prompt and the excerpt together are what produced the output. Keying on accession
    alone would serve predictions made against a different input -- the same stale-cache trap that
    src/labels/build.py guards against -- and would silently conflate the two prompt modes, whose
    whole purpose is to be compared against each other.
    """
    digest = hashlib.sha256((system_prompt + excerpt).encode()).hexdigest()[:12]
    return PREDICTION_CACHE / _slug(model_name) / f"{_slug(accession_no)}_{digest}.json"


def predict_dataset(
    examples: list[dict], model: Model, use_cache: bool = True, progress: bool = True
) -> tuple[list[str], dict]:
    """Raw prediction text per example, in order, plus a run summary.

    Failed generations become empty strings rather than gaps: the scorer treats them as invalid
    output, which is the honest reading -- a model that cannot answer got the filing wrong.
    """
    raws: list[str] = []
    stats = {"cached": 0, "generated": 0, "failed": 0, "truncated": 0, "refused": 0,
             "input_tokens": 0, "output_tokens": 0}

    for i, example in enumerate(examples, 1):
        excerpt = example["messages"][1]["content"]
        accession = example.get("accession_no", f"row{i}")
        path = cache_path(model.name, accession, excerpt, model.system_prompt)

        prediction: Prediction | None = None
        if use_cache and path.exists():
            try:
                prediction = Prediction.from_dict(json.loads(path.read_text()))
                stats["cached"] += 1
            except Exception as e:
                logger.warning("Ignoring corrupt prediction cache %s: %s", path.name, e)

        if prediction is None:
            if progress:
                logger.info("[%d/%d] %s", i, len(examples), example.get("ticker", accession))
            try:
                prediction = model.predict(excerpt)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(prediction.to_dict(), indent=2))
                stats["generated"] += 1
            except PredictionError as e:
                logger.error("  generation failed for %s: %s", example.get("ticker"), e)
                stats["failed"] += 1
                raws.append("")
                continue

        if prediction.truncated:
            logger.warning("  %s output was truncated at max_tokens", example.get("ticker"))
            stats["truncated"] += 1
        if prediction.refused:
            logger.warning("  %s was declined by a safety classifier", example.get("ticker"))
            stats["refused"] += 1
        stats["input_tokens"] += prediction.input_tokens or 0
        stats["output_tokens"] += prediction.output_tokens or 0
        raws.append(prediction.raw)

    return raws, stats


def estimate_cost(stats: dict, price_per_mtok: dict | None = None) -> float | None:
    """Dollar cost of a generation run, for the cost side of the model comparison.

    Only meaningful for API backends -- a locally served model's cost is GPU-hours, not tokens.
    """
    price = price_per_mtok or FRONTIER_PRICE_PER_MTOK
    if not stats.get("input_tokens") and not stats.get("output_tokens"):
        return None
    return (
        stats["input_tokens"] / 1e6 * price["input"]
        + stats["output_tokens"] / 1e6 * price["output"]
    )
