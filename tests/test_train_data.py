"""Unit tests for src/train/data.py.

A fake tokenizer keeps these offline and fast, and lets the tests assert on exact token positions
-- which is the only way to check loss masking, since a mis-masked run produces no error at all.
"""

from __future__ import annotations

import pytest

from src.train.data import (
    IGNORE_INDEX,
    PaddingCollator,
    encode_example,
    encode_split,
)


class FakeTokenizer:
    """Character-level tokenizer with a Llama-ish chat template. Ids are ord() so a decode is
    trivial and assertions can be made on the exact text the labels cover."""

    eos_token_id = 2
    pad_token_id = 0

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        rendered = "".join(f"<|{m['role']}|>{m['content']}" for m in messages)
        if add_generation_prompt:
            rendered += "<|assistant|>"
        return rendered

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [ord(c) for c in text]}

    def decode(self, ids):
        return "".join(chr(i) for i in ids if i not in (self.eos_token_id, self.pad_token_id))


TOK = FakeTokenizer()


def make_row(user="FILING TEXT", assistant='{"a":1}', ticker="TEST"):
    return {
        "ticker": ticker,
        "messages": [
            {"role": "system", "content": "SYS"},
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ],
    }


def test_loss_is_computed_only_on_the_assistant_turn():
    """The single most important property here: ~87% of each example is filing text the model is
    always given, and training on it wastes the run."""
    row = make_row()
    item, _ = encode_example(TOK, row, max_seq_len=512)

    supervised = [
        tok for tok, label in zip(item["input_ids"], item["labels"]) if label != IGNORE_INDEX
    ]
    assert TOK.decode(supervised) == '{"a":1}'

    # Everything before the target is masked, and nothing else is.
    n_masked = sum(1 for label in item["labels"] if label == IGNORE_INDEX)
    assert n_masked == len(item["input_ids"]) - len(supervised)
    assert "FILING TEXT" not in TOK.decode(supervised)


def test_labels_align_with_input_ids():
    item, _ = encode_example(TOK, make_row(), max_seq_len=512)
    for token, label in zip(item["input_ids"], item["labels"]):
        assert label in (IGNORE_INDEX, token)
    assert len(item["input_ids"]) == len(item["labels"]) == len(item["attention_mask"])


def test_eos_is_appended_to_the_target():
    """Without EOS the model never learns to stop and generation runs to max_new_tokens."""
    item, _ = encode_example(TOK, make_row(), max_seq_len=512)
    assert item["input_ids"][-1] == TOK.eos_token_id
    assert item["labels"][-1] == TOK.eos_token_id


def test_prompt_is_truncated_in_the_middle_not_the_ends():
    """Statements sit at the head of the excerpt and risk factors at the tail; both carry labels.
    MD&A in the middle carries none, so it is the only safe thing to drop."""
    user = "HEAD" + "M" * 400 + "TAIL"
    item, truncated = encode_example(TOK, make_row(user=user), max_seq_len=120)

    assert truncated
    prompt = TOK.decode([
        tok for tok, label in zip(item["input_ids"], item["labels"]) if label == IGNORE_INDEX
    ])
    assert "HEAD" in prompt and "TAIL" in prompt
    assert prompt.count("M") < 400  # the middle is what went


def test_truncation_never_touches_the_target():
    row = make_row(user="X" * 500, assistant='{"long":"target"}')
    item, truncated = encode_example(TOK, row, max_seq_len=100)

    assert truncated
    supervised = [
        tok for tok, label in zip(item["input_ids"], item["labels"]) if label != IGNORE_INDEX
    ]
    assert TOK.decode(supervised) == '{"long":"target"}'
    assert len(item["input_ids"]) <= 100


def test_target_larger_than_the_window_is_an_error_not_a_silent_cut():
    with pytest.raises(ValueError, match="does not fit"):
        encode_example(TOK, make_row(assistant="T" * 200), max_seq_len=50)


def test_encode_split_reports_the_supervised_share():
    rows = [make_row(user="U" * 100), make_row(user="U" * 100)]
    _, stats = encode_split(TOK, rows, max_seq_len=512)

    assert stats.n == 2
    assert stats.truncated == 0
    assert stats.completion_tokens < stats.prompt_tokens  # targets are the small part
    assert "supervised" in stats.summary()


def test_collator_pads_labels_with_ignore_index():
    """Padding labels with pad_token_id instead trains the model to emit padding."""
    # The collator returns tensors, so this is the one test that needs torch. Training runs on a
    # rented GPU box; the laptop keeps a lean venv, so skip rather than pull ~2.5GB of wheels.
    pytest.importorskip("torch")

    short, _ = encode_example(TOK, make_row(user="A"), max_seq_len=512)
    long, _ = encode_example(TOK, make_row(user="B" * 50), max_seq_len=512)

    batch = PaddingCollator(TOK)([short, long])

    assert batch["input_ids"].shape == batch["labels"].shape == batch["attention_mask"].shape
    assert batch["input_ids"].shape[1] % 8 == 0
    padded_row = batch["labels"][0].tolist()
    assert padded_row[-1] == IGNORE_INDEX
    assert batch["attention_mask"][0].tolist()[-1] == 0
