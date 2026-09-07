# Finetuning-EDGAR

[![Model on Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Llama--3.1--8B--edgar--10k--qlora-FFD21E)](https://huggingface.co/Sayar1106/Llama-3.1-8B-edgar-10k-qlora)
[![License: Llama 3.1](https://img.shields.io/badge/model%20license-Llama%203.1%20Community-2C5A88)](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct/blob/main/LICENSE)

Fine-tuning an open-source LLM (Llama 3.1 8B, QLoRA) to read SEC 10-K filings and extract
**structured financial data** — key financials, business segments, and categorized risk factors —
as validated JSON.

Numeric ground truth comes from SEC's **XBRL companyfacts API**, so the hardest labels (financial
figures) are free and exact rather than teacher-model generated. Qualitative fields (risk-factor
categories/summaries) are teacher-LLM labeled, and audited against a blind human sample —
`python -m src.labels.audit` draws the sheet, collects verdicts, and reports agreement with
filing-clustered intervals. All 60 sampled risks across 15 filings are now reviewed: **72% blind
category agreement** (76% random stratum, 58% test stratum), 100% grounded in the filing text, 100%
summary-faithful. Agreement rises to 86% if verdicts revised after seeing the teacher's answer are
counted as recorded — the blind figure is the one reported, and the gap is itself a finding
([D29](docs/decisions.md#d29--the-audit-reports-blind-agreement-not-the-recorded-rate)). Category
boundaries are documented in [`docs/taxonomy.md`](docs/taxonomy.md), reverse-engineered from the
teacher's own usage because the prompt never defined them
([D28](docs/decisions.md#d28--the-taxonomy-ships-to-the-teacher-undefined-and-the-audit-found-it)).

## Results

| Model                      | Schema-valid % | Numeric accuracy (exact or ±1%) | Risk-factor F1 |
|-----------------------------|:---:|:---:|:---:|
| Llama 3.1 8B (base, prose prompt) | 0 / 0 strict | 0.0 | 0.0 |
| Llama 3.1 8B (base, schema prompt) | 91.7 / 0 strict | 83.3 | 43.5 |
| **Llama 3.1 8B (fine-tuned)** | **100 / 100 strict** | **97.9** | **77.3** |
| Sonnet 5 (zero-shot, schema prompt) | 100 / 33.3 strict | 97.9 | 79.2 |

The fine-tuned row is the student under the prose prompt it was trained on; the base rows are the
same weights under both prompts ([D17](docs/decisions.md#d17--two-prompt-modes-both-published)).
The three Llama rows were run 2026-09-07 on one A100-SXM4-80GB; the Sonnet 5 row is an API run from
2026-08-02, with no temperature control available and extended thinking enabled, so it is not a
same-session comparison. All reports are checked in under `data/eval/reports/`.

**Strict schema validity is where fine-tuning wins outright: 0% → 100%.** The base model never once
emitted parseable JSON — only fence-and-prose stripping recovers 91.7% of it — and Sonnet manages 33.3%.
On the numbers themselves the student matches the frontier model (97.9 vs 97.9) and lands just under
it on risk F1 (77.3 vs 79.2), so the honest claim is *parity with Sonnet on content, a decisive win on
output discipline*, not a general accuracy win. Base numeric accuracy of 83.3% under the schema prompt
is the figure that keeps this honest: most of the numeric gain was already there before fine-tuning.

12 held-out companies, unconstrained decoding, $0.34. Run 2026-08-02; the full report is checked in
at `data/eval/reports/claude-sonnet-5_schema_test.json`. Schema-valid is reported after fence
stripping / strictly. Under the prose-only prompt the student trains on, the same model scores 0% —
see [D17](docs/decisions.md#d17--two-prompt-modes-both-published).

These figures move between runs: the API rejects a non-default `temperature`, so there is no seed to
pin. An earlier run of the same commit scored 25 / 95.8 / 76.7 — every one of those inside the
intervals below, and each gap is a single filing or a single field
([Q7](docs/decisions.md#q7--reproducibility-is-partial)).

**Every headline number carries a 95% bootstrap confidence interval**, resampled over filings rather
than field instances, because the four numeric fields within one filing are correlated. At n=12 the
interval is wide and the point estimate alone would invite a comparison the data cannot support — see
[D25](docs/decisions.md#d25--every-headline-metric-carries-a-bootstrap-interval).

**The test split contains no filing the base model could have memorized.** All 12 test filings are
FY2025 or FY2026, against a Llama 3.1 data cutoff of ~Dec 2023:

| Split | Filings | Companies | FY ≥ 2024 |
|---|---|---|---|
| train | 186 | 155 | 91.4% |
| val | 22 | 18 | 90.9% |
| **test** | **12** | **12** | **100%** |

Splits are assigned by hashing the ticker before fiscal year is known, so this is a property of the
corpus rather than a filter chosen to flatter the result
([Q3](docs/decisions.md#q3--how-do-you-know-the-base-model-hadnt-already-memorized-these-filings)).

## Project layout

```
src/
  data/     # EDGAR ingestion: download + parse filings into sections
  labels/   # XBRL alignment + teacher-LLM labeling -> training examples
  train/    # QLoRA fine-tuning (Unsloth)
  eval/     # extraction metrics + baseline/comparison runs
configs/    # training + eval YAML configs
tests/      # parser + metrics unit tests
docs/       # design notes, deferred ideas
```

## Setup

```bash
pip install -e ".[dev]"
cp .env.example .env   # fill in EDGAR_IDENTITY, ANTHROPIC_API_KEY, WANDB_API_KEY, HF_TOKEN
```

See extras in `pyproject.toml` for stage-specific dependencies (`labels`, `train`, `eval`, `demo`).

## Pipeline

1. `src/data/` — download and parse 10-K filings for ~200 companies.
2. `src/labels/` — build the labeled extraction dataset (XBRL + teacher LLM), company-level splits,
   and audit the teacher's labels by hand (`src/labels/audit.py`, worksheet in `docs/audit/`).
3. `src/eval/` — extraction metrics and baseline runs.
4. `src/train/` — QLoRA fine-tuning on a rented GPU.
5. The adapter is pushed to the Hugging Face Hub as `Sayar1106/Llama-3.1-8B-edgar-10k-qlora`;
   eval reports for every run are checked in under `data/eval/reports/`, and the model card is at
   [`docs/model_card.md`](docs/model_card.md).

A side-by-side of what the base and fine-tuned models actually emit for one filing, with the raw
outputs and the full per-filing table:
**[view the demo](https://sayar1106.github.io/Finetuning-EDGAR/demo.html)**
(source: [`docs/demo.html`](docs/demo.html) — a static page, no network or GPU behind it).

Module-level data flow, caches, and where each artifact comes from:
[`docs/architecture.md`](docs/architecture.md).

Every non-obvious choice, with the evidence behind it:
[`docs/decisions.md`](docs/decisions.md). Run logs from the GPU session:
[`docs/runs/`](docs/runs/).

## License and attribution

Built with Llama.

The fine-tuned adapter is a derivative of Llama 3.1 and is governed by the
[Llama 3.1 Community License](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct/blob/main/LICENSE);
use must comply with the [Acceptable Use Policy](https://llama.meta.com/llama3_1/use-policy). The
attribution notice the Agreement requires is in [`NOTICE`](NOTICE), and the obligations are itemised
in [Q9](docs/decisions.md#q9--licensing-of-the-published-artifact). This repository's own code is
MIT ([`LICENSE`](LICENSE)); SEC filing content is public domain.

## Design decisions

[`docs/decisions.md`](docs/decisions.md) records every non-obvious choice with its rationale and the
evidence behind it — why XBRL supplies the numeric labels, why the teacher labels the excerpt rather
than the full filing, why the frontier baseline is Sonnet and deliberately not the model that
generated the gold labels. It also carries an incident log (the XBRL fact-selection bug that
corrupted 34 of 182 filings' labels, and the macOS low-disk purge that ate the repo).

## Future work

`docs/future-work.md` records ideas scoped out of v1 and why — notably **fund strategy-compliance
checking**: extracting a fund's stated investment policy from its prospectus and verifying it
against actual N-PORT holdings. It includes the spike evidence and the blocker (silent series
misalignment in fund filings) needed to pick it up later.
