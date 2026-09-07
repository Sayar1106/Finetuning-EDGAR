---
base_model: meta-llama/Llama-3.1-8B-Instruct
library_name: peft
license: llama3.1
pipeline_tag: text-generation
tags:
  - lora
  - qlora
  - peft
  - information-extraction
  - sec-filings
  - xbrl
---

# Llama-3.1-8B-edgar-10k-qlora

Built with Llama.

A QLoRA adapter that pulls structured JSON out of SEC 10-K filings: four financial figures and a
categorized list of risk factors.

## What it does

Given the MD&A/financials excerpt and the Item 1A risk-factor text of a 10-K, it emits:

```json
{
  "financials": {"revenue": 0, "net_income": 0, "total_assets": 0, "eps_diluted": 0.0},
  "risk_factors": [{"title": "...", "category": "...", "summary": "..."}]
}
```

`category` is one of 12 values defined in [`docs/taxonomy.md`](taxonomy.md).

## Intended use

Bulk pre-extraction where a human checks the output: building a dataset, triaging a filing set, or
drafting a structured summary for review.

Not for investment decisions or anything where a wrong number is costly: numeric accuracy is 97.9%
on twelve filings, and the model gives no signal when it is wrong. It has only seen 10-Ks, so 10-Qs,
8-Ks, S-1s and non-SEC financial text are untested, and filings that satisfy Item 1A by
cross-reference carry no risk text to work from.

For the four financial figures, XBRL companyfacts are exact and free; use those. This model is for
the qualitative half, where no structured source exists.

## Results

Twelve held-out companies, unconstrained decoding. Every figure carries a 95% percentile bootstrap
confidence interval resampled over filings, since numeric fields within one filing are correlated.

| Model | Prompt | Schema valid (strict / lenient) | Numeric accuracy | Risk-factor F1 |
|---|---|:---:|:---:|:---:|
| Llama 3.1 8B base | prose | 0.0% / 0.0% | 0.0% | 0.0% |
| Llama 3.1 8B base | schema | 0.0% / 91.7% | 83.3% | 43.5% |
| **This adapter** | **prose** | **100% / 100%** | **97.9%** | **77.3%** |
| This adapter | schema | 100% / 100% | 93.8% | 82.0% |
| Claude Sonnet 5 | schema | 33.3% / 100% | 97.9% | 79.2% |

The rows are not interchangeable. The four Llama rows ran 2026-09-07 on one A100-SXM4-80GB against
stock `meta-llama/Llama-3.1-8B-Instruct` weights; the Sonnet 5 row is an API run from 2026-08-02
with no temperature control and extended thinking enabled, so it is not a same-hardware comparison.
All five reports are under `data/eval/reports/`.

Numeric accuracy counts a value as correct if it is exact or within 1% (`CLOSE_TOL` in
`src/eval/metrics.py`), which covers a filing that writes `22.2` where XBRL holds `22.18`. Under the
prose prompt, revenue is 9 exact and 3 close.

Risk-factor F1 is macro-averaged category F1 over filings. A stricter metric that also requires each
risk to be matched to the right gold risk gives 70.1% for this adapter and 75.3% for Sonnet,
reversing the ordering above. Both metrics are in the checked-in reports.

### Reading the table

Fine-tuning mostly bought output discipline. The base model never emitted parseable JSON at all: 0%
strict on both prompts, and only fence-and-prose stripping recovers 91.7% of the schema run. Sonnet
manages 33.3% strict; this adapter is at 100%. No schema failure occurred in twelve filings, so the
[100%, 100%] interval is a boundary artifact of the percentile bootstrap, not evidence that the true
failure rate is zero.

Risk categorization improved substantially, 43.5% to 77.3% macro F1. Numeric extraction barely
moved: the base model already reaches 83.3% once handed the schema, so most of the numeric headline
predates fine-tuning, and a table that hid the base-schema row would be overselling.

Against the frontier model that is parity on content (97.9% vs 97.9% numeric, 77.3% vs 79.2%
category F1) and a decisive win on output discipline (100% vs 33.3% strict).

## Cost

Measured, and it retracts an earlier claim from this project. Generation took 696s for twelve
filings, 58.0s each and ~99 tok/s end to end, on an A100-SXM4-80GB at $1.59/hr:

| | per filing |
|---|---|
| This adapter, single-stream on a rented A100 | **$0.0256** |
| Claude Sonnet 5 (measured, same twelve filings) | **$0.0283** |

That is 0.91x: parity, not the ~1/20th an earlier version of this project claimed. It is the
generous reading, counting generation only and excluding the model load the API never pays for.
Reaching a twentieth needs roughly 18x more throughput, meaning batched inference this project has
not built. An hourly GPU is only cheap at high utilization; at twelve filings decoded one at a time,
the API wins.

## Training

| | |
|---|---|
| Base at training | `unsloth/llama-3.1-8b-instruct-unsloth-bnb-4bit` (4-bit NF4, double quant) |
| Base at evaluation | `meta-llama/Llama-3.1-8B-Instruct`, for every number above |
| Method | QLoRA via Unsloth; r=32, alpha=64, dropout=0.05 |
| Target modules | `q_proj` `k_proj` `v_proj` `o_proj` `gate_proj` `up_proj` `down_proj` |
| Sequence length | 12,288, with 0 of 220 examples truncated |
| Schedule | 3 epochs configured, lr 2e-4 cosine, effective batch 8, `paged_adamw_8bit` |
| Hardware | 1x A100-SXM4-80GB, 20.3 min |

The train and eval bases differ deliberately. Training used Unsloth's 4-bit repack for speed, while
evaluation ran the adapter on stock Meta weights so the reported numbers are not contingent on a
third-party requantization. `modules_to_save` is null and only the seven standard projection modules
are adapted, so serving on stock weights requires no merge.

### The shipped weights are epoch 2, not epoch 3

Validation loss turned while training loss kept falling:

| epoch | eval_loss |
|---|---|
| 1 | 0.4583 |
| 2 | **0.4459** |
| 3 | 0.4709 |

`load_best_model_at_end` with `metric_for_best_model: eval_loss` restored `checkpoint-48` before
saving, so the `num_train_epochs: 3` in the config does not describe what shipped. Ordinary
overfitting on 186 examples.

## Training data

220 10-K filings from 185 US public companies, FY2023 to FY2025, split by company so that no company
appears in two splits: 186 train, 22 validation, 12 test.

The four numeric fields are exact: SEC XBRL companyfacts, the numbers the company itself filed, not
model output. There is no label noise to speak of.

Risk-factor titles, categories and summaries are teacher-generated by Claude Haiku 4.5, audited
against a blind human sample of 60 risks across 15 filings. Blind agreement was 72% (76% on the
random stratum, 58% on the test stratum, n=3 filings there); grounding and summary faithfulness were
both 100%. So the risk-category ceiling this model was trained toward is itself around 72-76%
agreement with a careful human, and its 77.3% F1 should be read against that, not against
perfection.

About 84% of each training example is filing text and 16% is the supervised target (180,182
supervised of 1,127,859 total tokens).

## Limitations

- Twelve test filings. Confidence intervals are wide, and stated everywhere.
- The teacher sets the ceiling for risk categorization, at ~72% blind human agreement.
- Risk-factor F1 matches predicted risks to gold risks by lexical overlap, which favors models that
  reuse the filing's wording.
- One training run and one seed, so no variance estimate.
- FY2023 to FY2025 US large-caps only. Smaller filers and older filings are untested.

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE = "meta-llama/Llama-3.1-8B-Instruct"
ADAPTER = "Sayar1106/Llama-3.1-8B-edgar-10k-qlora"

tokenizer = AutoTokenizer.from_pretrained(BASE)
model = AutoModelForCausalLM.from_pretrained(BASE, device_map="auto", dtype="bfloat16")
model = PeftModel.from_pretrained(model, ADAPTER)

messages = [
    {"role": "system", "content": SYSTEM_PROMPT},   # the prose prompt from src/eval/predict.py
    {"role": "user", "content": filing_excerpt},
]
inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
out = model.generate(inputs.to(model.device), max_new_tokens=2048, do_sample=False)
print(tokenizer.decode(out[0][inputs.shape[-1]:], skip_special_tokens=True))
```

`SYSTEM_PROMPT` is the prose prompt from `PROMPT_MODES["trained"]`, not a schema-bearing one; the
headline row is that prompt. Input averages 4,912 tokens per filing and output 805 across the twelve
test filings.

The snippet is untested here: the development machine deliberately carries no `torch` and no
weights. It uses the same API as the eval harness.

## License

This is a derivative of Llama 3.1 and is governed by the
[Llama 3.1 Community License](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct/blob/main/LICENSE),
Copyright © Meta Platforms, Inc. Use must comply with the
[Acceptable Use Policy](https://llama.meta.com/llama3_1/use-policy).

Required attribution notice, reproduced verbatim from the Agreement:

> Llama 3.1 is licensed under the Llama 3.1 Community License, Copyright © Meta Platforms, Inc. All Rights Reserved.

The repository name carries the `Llama` prefix the Agreement requires of derivative model names, and
a copy of the Agreement is distributed alongside these weights.

The project's own code is MIT; SEC filing content is public domain. Neither changes the terms above.

## Citation

```bibtex
@misc{banerjee2026edgarqlora,
  title  = {Llama-3.1-8B-edgar-10k-qlora: structured extraction from SEC 10-K filings},
  author = {Banerjee, Sayar},
  year   = {2026},
  url    = {https://huggingface.co/Sayar1106/Llama-3.1-8B-edgar-10k-qlora}
}
```
