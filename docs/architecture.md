# Architecture

End-to-end flow from SEC filings to a scored extraction model. Every box is a module that exists in
the repo; every cylinder is an artifact on disk. Rationale for the labeled edges lives in
[`decisions.md`](decisions.md).

```mermaid
flowchart TB
    EDGAR[/"SEC EDGAR<br/>10-K documents + XBRL companyfacts API"/]
    HAIKU[/"Claude Haiku 4.5<br/>teacher"/]
    SONNET[/"Claude Sonnet 5<br/>frontier baseline"/]
    HUB[/"Hugging Face<br/>Llama 3.1 8B Instruct"/]

    subgraph ING["1 · Ingest — src/data/"]
        direction LR
        CO["companies.py<br/>~200 pinned tickers"]
        DL["download.py<br/>edgartools → Item 1 / 1A / 7"]
        ST["statements.py<br/>Item 8 → compact text"]
        XB["xbrl_facts.py<br/>5 numeric fields, exact"]
        CO --> DL
        DL --> ST
        DL --> XB
    end

    RAW[("data/raw/<br/>one JSON per filing<br/>FilingSections + FinancialFacts")]

    subgraph LAB["2 · Label — src/labels/"]
        direction LR
        EX["excerpt.py<br/>bounded excerpt<br/>+ check_grounding"]
        TE["teacher.py<br/>risk category + summary only"]
        SP["splits.py<br/>sha256 ticker → split"]
        BU["build.py<br/>chat-format examples"]
        EX --> TE
        TE --> BU
        SP --> BU
    end

    TCACHE[("_teacher_cache/<br/>keyed on excerpt digest")]
    PROC[("data/processed/<br/>train 186 · val 22 · test 12")]

    subgraph AUD["Audit — src/labels/audit.py"]
        direction LR
        A1["sample<br/>two-stage cluster sample"]
        A2["review<br/>blind category elicitation"]
        A3["score<br/>agreement + clustered CI"]
        A1 --> A2
        A2 --> A3
    end

    subgraph TR["3 · Train — src/train/"]
        direction LR
        CFG["configs/sft_llama31_8b.yaml<br/>r=32 · 3 epochs · 4-bit NF4"]
        ME["measure.py<br/>token distribution → max_seq_len"]
        TD["data.py<br/>loss mask + middle truncation"]
        SFT["sft.py<br/>QLoRA · Unsloth, peft fallback"]
        ME --> TD
        TD --> SFT
        CFG --> SFT
    end

    ADAPTER[("LoRA adapter<br/>+ resolved config snapshot")]

    subgraph EV["4 · Evaluate — src/eval/"]
        direction LR
        RN["run.py"]
        PR["predict.py<br/>backends: anthropic · hf · hf+adapter<br/>prompt modes: trained · schema"]
        MT["metrics.py<br/>schema validity · numeric · risk F1<br/>cluster bootstrap CIs"]
        RN --> PR
        PR --> MT
    end

    PCACHE[("predictions/<br/>keyed on prompt+excerpt digest")]
    REPORTS[("data/eval/reports/*.json<br/>→ README results table")]
    APP["5 · Ship — app/<br/>Gradio demo + model card"]

    EDGAR --> ING
    ING --> RAW
    RAW --> EX
    HAIKU --> TE
    TE --> TCACHE
    TCACHE --> BU
    BU --> PROC
    PROC -->|"train + val"| TD
    PROC -->|"sampled risks"| A1
    PROC -->|"test split, 12 filings"| PR
    HUB --> SFT
    SFT --> ADAPTER
    ADAPTER -->|"fine-tuned run"| PR
    HUB -->|"base run"| PR
    SONNET -->|"frontier run"| PR
    PR --> PCACHE
    MT --> REPORTS
    A3 -.->|"conditions every risk-factor number"| REPORTS
    ADAPTER --> APP

    classDef pending stroke-dasharray:4,stroke-width:1px
    class APP pending
```

Dashed = not yet produced. As of the 2026-09-07 A100 run the adapter exists (pushed to the Hub;
`docs/runs/finetune.log`) and the fine-tuned eval rows are checked in under `data/eval/reports/`
(`docs/runs/fteval.log`); only the ship step -- demo and a written model card -- is still dashed.

## What the shape encodes

**Two sources of truth, deliberately split.** Numeric labels come from XBRL and cost nothing; only
risk categories and summaries reach a teacher model. That is the one paid edge in labeling, and it is
also the only edge whose output needs auditing — hence `audit.py` hanging off `data/processed/`
rather than off the numerics ([D2](decisions.md#d2--xbrl-companyfacts-as-programmatic-ground-truth-for-numerics),
[D3](decisions.md#d3--teacher-model-only-for-the-qualitative-half)).

**The excerpt is upstream of the teacher, not parallel to it.** `excerpt.py` runs first and
`teacher.py` labels *that text*, so every risk factor in a target is present in the input the student
sees ([D7](decisions.md#d7--the-teacher-labels-the-excerpt-not-the-full-filing)).

**Splitting happens before labeling, on the ticker.** `splits.py` feeds `build.py`, and assignment is
a hash of the ticker computed before fiscal year is known — so the test set's composition is a
property of the corpus, not a filter ([D5](decisions.md#d5--split-by-company-never-by-filing)).

**One scorer, three models.** `predict.py` returns raw text unmodified and `metrics.py` does all
parsing, so schema validity stays measurable and the base, fine-tuned, and frontier runs are scored
by identical code with no constrained decoding for anyone
([D16](decisions.md#d16--the-baseline-is-not-handicapped)).

**Both caches key on a content digest.** The teacher cache keys on the excerpt, the prediction cache
on prompt + excerpt. Changing the example format never re-pays for labeling; changing the scorer
never re-pays for generation. This is also why `refresh_facts.py` rewrites only the `financials`
block and leaves section text byte-identical ([D23](decisions.md#d23--teacher-labels-are-cached-per-filing)).
