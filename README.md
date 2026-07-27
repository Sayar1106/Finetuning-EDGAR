# Finetuning-EDGAR

Fine-tuning an open-source LLM (Llama 3.1 8B, QLoRA) to read SEC 10-K filings and extract
**structured financial data** — key financials, business segments, and categorized risk factors —
as validated JSON.

Numeric ground truth comes from SEC's **XBRL companyfacts API**, so the hardest labels (financial
figures) are free and exact rather than teacher-model generated. Qualitative fields (risk-factor
categories/summaries) are teacher-LLM labeled and spot-checked by hand.

## Results

_Fine-tuned rows populated after training._

| Model                      | Schema-valid % | Numeric exact-match | Risk-factor F1 |
|-----------------------------|:---:|:---:|:---:|
| Llama 3.1 8B (base, few-shot) | — | — | — |
| Llama 3.1 8B (fine-tuned)     | — | — | — |
| Sonnet 5 (zero-shot, schema prompt) | 100 / 25 strict | 95.8 | 76.7 |

12 held-out companies, unconstrained decoding, $0.71. Schema-valid is reported after fence
stripping / strictly. Under the prose-only prompt the student trains on, the same model scores 0% —
see [D17](docs/decisions.md#d17--two-prompt-modes-both-published).

## Project layout

```
src/
  data/     # EDGAR ingestion: download + parse filings into sections
  labels/   # XBRL alignment + teacher-LLM labeling -> training examples
  train/    # QLoRA fine-tuning (Unsloth)
  eval/     # extraction metrics + baseline/comparison runs
configs/    # training + eval YAML configs
notebooks/  # EDA, error analysis
app/        # Gradio demo
tests/      # parser + metrics unit tests
docs/       # design notes, deferred ideas
```

## Setup

```bash
pip install -e ".[dev]"
cp .env.example .env   # fill in EDGAR_IDENTITY, OPENAI_API_KEY, WANDB_API_KEY, HF_TOKEN
```

See extras in `pyproject.toml` for stage-specific dependencies (`labels`, `train`, `eval`, `demo`).

## Pipeline

1. `src/data/` — download and parse 10-K filings for ~200 companies.
2. `src/labels/` — build the labeled extraction dataset (XBRL + teacher LLM), company-level splits.
3. `src/eval/` — extraction metrics and baseline runs.
4. `src/train/` — QLoRA fine-tuning on a rented GPU.
5. `app/` — Gradio demo; model + model card published to Hugging Face Hub.

Full plan and rationale: see the project plan.

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
