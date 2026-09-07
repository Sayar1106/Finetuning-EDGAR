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

A QLoRA adapter that extracts structured JSON — four financial figures and a categorised list of
risk factors — from the text of SEC 10-K filings.

The point of the project is not the adapter. It is that every number below is reproducible from a
checked-in artifact, and that the claims which did not survive measurement were retracted rather
than softened. Two of them are recorded here.

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

Bulk pre-extraction from 10-K filings where the output is checked before use — building a dataset,
triaging a filing set, or drafting a structured summary a human reviews. It is a labour-saving
device for a reader who would otherwise parse the filing by hand.

## Out of scope

- **Investment decisions, or any use where a wrong number is costly.** Numeric accuracy is 97.9% on
  twelve filings, which means errors happen and the model does not signal them.
- **Anything but a 10-K.** Trained only on 10-Ks; 10-Qs, 8-Ks, S-1s and non-SEC financial text are
  untested.
- **Filings that satisfy Item 1A by cross-reference.** These contain no risk text; the training data
  flags them explicitly and the model has no way to.
- **A source of truth for financial figures.** XBRL companyfacts are exact and free. Use those. This
  model exists for the qualitative half, where no structured source exists.

## Results

Twelve held-out companies, unconstrained decoding. Every figure carries a 95% percentile bootstrap
confidence interval resampled over filings, because numeric fields within one filing are correlated.

| Model | Prompt | Schema valid (strict / lenient) | Numeric accuracy | Risk-factor F1 |
|---|---|:---:|:---:|:---:|
| Llama 3.1 8B base | prose | 0.0% / 0.0% | 0.0% | 0.0% |
| Llama 3.1 8B base | schema | 0.0% / 91.7% | 83.3% | 43.5% |
| **This adapter** | **prose** | **100% / 100%** | **97.9%** | **77.3%** |
| This adapter | schema | 100% / 100% | 93.8% | 82.0% |
| Claude Sonnet 5 | schema | 33.3% / 100% | 97.9% | 79.2% |

**Provenance differs by row and is not interchangeable.** The four Llama rows were generated
2026-09-07 on one A100-SXM4-80GB against stock `meta-llama/Llama-3.1-8B-Instruct` weights. The
Sonnet 5 row is an API run from 2026-08-02, with no temperature control available and extended
thinking enabled — it is not a same-session, same-hardware comparison and should not be read as one.
All five reports are checked in under `data/eval/reports/`.

**Numeric accuracy counts a value as correct if it is exact or within 1%** (`CLOSE_TOL` in
`src/eval/metrics.py`), which absorbs a filing that writes `22.2` where XBRL holds `22.18`. For this
adapter under the prose prompt, revenue is 9 exact and 3 close.

**Risk-factor F1 is macro-averaged category F1 over filings.** A stricter lower-bound metric that
also requires the risk to be matched to the right gold risk gives 70.1% for this adapter and 75.3%
for Sonnet — that ordering reverses relative to the table above, and both metrics are in the
checked-in reports.

### What the numbers mean

Fine-tuning bought two things and not a third.

**Output discipline, decisively.** The base model never once emitted parseable JSON — 0% strict
across both prompts, and only fence-and-prose stripping recovers 91.7% of the schema-prompt run.
Sonnet manages 33.3% strict. This adapter is at 100%. No schema failure was observed in twelve
filings; the [100%, 100%] interval is a boundary artifact of the percentile bootstrap, not evidence
that the true failure rate is zero.

**Risk categorisation, substantially.** 43.5% → 77.3% macro F1.

**Numeric extraction, barely.** The base model already reaches 83.3% when handed the schema. Most of
the numeric headline predates fine-tuning, and a table that hid the base-schema row would be
overselling.

Against the frontier model the honest reading is **parity on content** (97.9% vs 97.9% numeric,
77.3% vs 79.2% category F1) **and a decisive win on output discipline** (100% vs 33.3% strict).

## Cost

Measured, and it retracts an earlier claim. Generation took 696s for twelve filings — 58.0s each,
~99 tok/s end to end — on an A100-SXM4-80GB at $1.59/hr:

| | per filing |
|---|---|
| This adapter, single-stream on a rented A100 | **$0.0256** |
| Claude Sonnet 5 (measured, same twelve filings) | **$0.0283** |

That is **0.91×: parity, not the ~1/20th an earlier version of this project claimed.** The generous
reading, even — it counts generation only and excludes model loading, which the API never pays.
Reaching a twentieth would need roughly 18× more throughput, which means batched or
continuous-batching inference this project has not built and therefore does not claim.

A rented GPU billed by the hour is only cheap at high utilisation. At twelve filings decoded one at
a time, the frontier API wins.

## Training

| | |
|---|---|
| Base at training | `unsloth/llama-3.1-8b-instruct-unsloth-bnb-4bit` (4-bit NF4, double quant) |
| Base at evaluation | `meta-llama/Llama-3.1-8B-Instruct` — every number above |
| Method | QLoRA via Unsloth; r=32, α=64, dropout=0.05 |
| Target modules | `q_proj` `k_proj` `v_proj` `o_proj` `gate_proj` `up_proj` `down_proj` |
| Sequence length | 12,288 — 0 of 220 examples truncated |
| Schedule | 3 epochs configured, lr 2e-4 cosine, effective batch 8, `paged_adamw_8bit` |
| Hardware | 1× A100-SXM4-80GB, 20.3 min |

The train/eval base differs and that is deliberate: training used Unsloth's 4-bit repack for speed,
evaluation ran the adapter on stock Meta weights so the reported numbers are not contingent on a
third-party requantisation. `modules_to_save` is null and only the seven standard projection modules
are adapted, so no merge is required to serve it on stock weights.

### The shipped weights are epoch 2, not epoch 3

Validation loss turned while training loss kept falling:

| epoch | eval_loss |
|---|---|
| 1 | 0.4583 |
| 2 | **0.4459** |
| 3 | 0.4709 |

`load_best_model_at_end` with `metric_for_best_model: eval_loss` restored `checkpoint-48` before
saving. **Reading `num_train_epochs: 3` and assuming you have the epoch-3 model would be wrong.**
Ordinary overfitting on 186 examples; the third epoch is what produces the evidence that two is
right.

## Training data

220 10-K filings from 185 US public companies, FY2023–FY2025, split by company so no company appears
in two splits: **186 train / 22 validation / 12 test**.

Labels come from two sources with very different reliability, and the difference matters:

- **The four numeric fields are exact.** They come from SEC XBRL companyfacts — the same numbers the
  company filed — not from a model. There is no label noise to speak of.
- **Risk-factor titles, categories and summaries are teacher-generated** by Claude Haiku 4.5, and
  were audited against a blind human sample: 60 risks across 15 filings. **Blind agreement with the
  human reviewer was 72%** (76% on the random stratum, 58% on the test stratum, n=3 filings there).
  Grounding and summary faithfulness were 100%. So the risk-category ceiling this model was trained
  toward is itself around 72–76% agreement with a careful human, and its 77.3% F1 should be read
  against that ceiling rather than against perfection.

~84% of each training example is filing text and ~16% is the supervised target (measured:
180,182 supervised of 1,127,859 total tokens).

## Limitations

- **n=12.** The test set is twelve filings. Confidence intervals are wide and stated everywhere.
- **The teacher is the ceiling** for risk categorisation, at ~72% blind human agreement.
- **Risk-factor F1 rests on lexical overlap** for matching predicted to gold risks, which flatters
  models that reuse the filing's wording.
- **Single seed.** One training run; no variance estimate across seeds.
- **Time-boxed corpus.** FY2023–FY2025 US large-caps. Smaller filers and older filings are untested.

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

Use the **prose** system prompt the adapter was trained on (`PROMPT_MODES["trained"]` in
`src/eval/predict.py`), not a schema-bearing one — the headline row is that prompt. Input averages
4,912 tokens per filing and output 805, measured over the twelve test filings in trained mode.

*This snippet has not been executed in this repository's environment: the development machine
deliberately carries no `torch` and no model weights. It is written against the same API the eval
harness uses.*

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
