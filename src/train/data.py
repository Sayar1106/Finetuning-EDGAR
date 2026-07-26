"""Turn the labeled JSONL splits into tokenized training tensors.

Two things here decide whether a run learns anything, and both fail silently if wrong.

**Loss masking.** Every example is ~87% filing text and ~13% target JSON. Training on the whole
sequence spends almost all of the gradient signal teaching the model to reproduce 10-K prose it
will always be *given* at inference. Prompt positions are set to -100 so loss is computed only on
the assistant turn.

Masking is done by tokenizing the prompt and the completion separately and concatenating, rather
than by string-matching a response template in the rendered text. Template matching is the usual
approach and it degrades quietly: if the tokenizer renders the assistant header even slightly
differently than the literal you searched for, nothing matches, every position stays unmasked, and
the run looks fine while training on the wrong objective.

**Truncation.** The longest examples run past a comfortable sequence length. Truncating from the
end would cut the target itself; truncating from the front would cut the financial statements the
numeric labels come from. Both would leave labels the input no longer supports -- the same
ungrounded-label failure src/labels/excerpt.py exists to prevent. So an over-long prompt is
truncated in the *middle*, which is MD&A: the one section no label is derived from.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = ROOT / "data" / "processed"

IGNORE_INDEX = -100


@dataclass
class EncodingStats:
    n: int = 0
    truncated: int = 0
    max_tokens: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0

    def summary(self) -> str:
        share = self.completion_tokens / max(1, self.prompt_tokens + self.completion_tokens)
        return (
            f"{self.n} examples | longest {self.max_tokens} tokens | {self.truncated} truncated | "
            f"supervised {self.completion_tokens:,} of "
            f"{self.prompt_tokens + self.completion_tokens:,} tokens ({share:.1%})"
        )


def load_jsonl(path: Path | str) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def render_prompt(tokenizer, messages: list[dict]) -> str:
    """System + user turns, ending with the assistant header the model must continue from."""
    return tokenizer.apply_chat_template(
        [m for m in messages if m["role"] != "assistant"],
        tokenize=False,
        add_generation_prompt=True,
    )


def _truncate_middle(ids: list[int], budget: int) -> list[int]:
    """Keep the head and tail of `ids`, dropping the middle.

    The excerpt is ordered statements -> MD&A -> risk factors. Numeric labels come from the head
    and risk-factor labels from the tail, so the middle is the only part that can go without
    orphaning a label.
    """
    if len(ids) <= budget:
        return ids
    head = budget // 2
    tail = budget - head
    return ids[:head] + ids[len(ids) - tail :]


def encode_example(tokenizer, row: dict, max_seq_len: int) -> tuple[dict, bool]:
    """One row -> input_ids/labels/attention_mask, plus whether the prompt had to be truncated."""
    messages = row["messages"]
    assistant = next(m["content"] for m in messages if m["role"] == "assistant")

    prompt_ids = tokenizer(render_prompt(tokenizer, messages), add_special_tokens=False)["input_ids"]
    completion_ids = tokenizer(assistant, add_special_tokens=False)["input_ids"]
    if tokenizer.eos_token_id is not None:
        # Without EOS the model never learns to stop, and generation runs to max_new_tokens.
        completion_ids = completion_ids + [tokenizer.eos_token_id]

    # The completion is never truncated: it is the entire training signal.
    budget = max_seq_len - len(completion_ids)
    if budget <= 0:
        raise ValueError(
            f"{row.get('ticker')}: target alone is {len(completion_ids)} tokens, which does not "
            f"fit in max_seq_len={max_seq_len}"
        )
    truncated = len(prompt_ids) > budget
    prompt_ids = _truncate_middle(prompt_ids, budget)

    input_ids = prompt_ids + completion_ids
    labels = [IGNORE_INDEX] * len(prompt_ids) + completion_ids
    return (
        {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": [1] * len(input_ids),
        },
        truncated,
    )


def encode_split(tokenizer, rows: list[dict], max_seq_len: int) -> tuple[list[dict], EncodingStats]:
    encoded, stats = [], EncodingStats()
    for row in rows:
        item, truncated = encode_example(tokenizer, row, max_seq_len)
        encoded.append(item)
        stats.n += 1
        stats.truncated += int(truncated)
        stats.max_tokens = max(stats.max_tokens, len(item["input_ids"]))
        supervised = sum(1 for label in item["labels"] if label != IGNORE_INDEX)
        stats.completion_tokens += supervised
        stats.prompt_tokens += len(item["input_ids"]) - supervised
        if truncated:
            logger.warning("%s: prompt truncated to fit max_seq_len=%d", row.get("ticker"), max_seq_len)
    return encoded, stats


def build_dataset(
    tokenizer,
    split: str,
    max_seq_len: int,
    data_dir: Path | None = None,
    limit: int | None = None,
):
    """An HF Dataset of tokenized examples for `split`, plus the encoding stats.

    `limit` subsets the *rows* before encoding, so a smoke run does not pay to tokenize the whole
    split -- or trip over a long example it was never going to train on.
    """
    from datasets import Dataset

    rows = load_jsonl((data_dir or PROCESSED_DIR) / f"{split}.jsonl")
    if limit:
        rows = rows[:limit]
    encoded, stats = encode_split(tokenizer, rows, max_seq_len)
    logger.info("%s: %s", split, stats.summary())
    return Dataset.from_list(encoded), stats


@dataclass
class PaddingCollator:
    """Pads a batch and keeps padded label positions masked.

    `DataCollatorForSeq2Seq` would also work, but writing it out keeps the -100 handling visible --
    padding labels with `pad_token_id` instead of IGNORE_INDEX trains the model to emit padding.
    """

    tokenizer: object
    pad_to_multiple_of: int = 8

    def __call__(self, features: list[dict]) -> dict:
        import torch

        longest = max(len(f["input_ids"]) for f in features)
        if self.pad_to_multiple_of:
            longest = -(-longest // self.pad_to_multiple_of) * self.pad_to_multiple_of

        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.tokenizer.eos_token_id

        batch = {"input_ids": [], "labels": [], "attention_mask": []}
        for f in features:
            pad = longest - len(f["input_ids"])
            batch["input_ids"].append(f["input_ids"] + [pad_id] * pad)
            batch["labels"].append(f["labels"] + [IGNORE_INDEX] * pad)
            batch["attention_mask"].append(f["attention_mask"] + [0] * pad)
        return {k: torch.tensor(v, dtype=torch.long) for k, v in batch.items()}
