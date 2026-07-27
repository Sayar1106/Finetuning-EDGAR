# Decision log

Why the project is built the way it is. Each entry is a choice that had a real alternative, the
reason it went the way it did, and the evidence behind it. Backfilled for Days 1–5 on 2026-07-26;
appended to as work continues.

Companion docs: [`future-work.md`](future-work.md) for scoped-out ideas, README for results.

---

## 1. Task and ground truth

### D1 — Structured extraction from 10-Ks, not summarization or Q&A
The task had to be **objectively evaluable**. Extraction into a fixed JSON schema scores as exact
match and F1; summarization scores as an LLM judging another LLM. Eval rigor is the point of the
project, so anything requiring a judge model was out.

### D2 — XBRL companyfacts as programmatic ground truth for numerics
SEC filers submit machine-readable XBRL alongside the human-readable 10-K. That makes every
financial figure a **free, exact, non-teacher-generated label**. The expensive half of labeling
disappears, and the hardest fields are the ones with the strongest ground truth.

*Alternative rejected:* teacher-LLM labels for numerics too — cheaper to build, but then the
headline metric measures agreement with another model rather than agreement with the filed truth.

### D3 — Teacher model only for the qualitative half
Risk-factor categories and summaries have no programmatic source, so Claude Haiku 4.5 labels them
(`src/labels/teacher.py`). High-volume short-output classification over a fixed taxonomy — per-call
cost dominates, and the task is well within a small model's range. `messages.parse` constrains the
output to the schema, so there is no JSON-repair path to maintain.

The `Teacher` protocol keeps the provider swappable: `src/labels/build.py` depends on the protocol,
not on Anthropic.

### D4 — Closed risk-factor taxonomy
A fixed category set (`src/labels/schema.py:16`) makes risk categorization scoreable as multi-label
F1. Open-vocabulary categories would collapse the metric into fuzzy string matching.

---

## 2. Dataset construction

### D5 — Split by company, never by filing
A company files near-identical boilerplate year over year. Splitting by filing would put FY2024
Apple in train and FY2025 Apple in test, and the resulting score would measure memorization of
Apple's phrasing. Assignment is a deterministic hash of the ticker, so adding companies never
reshuffles existing ones. Current split: **155 train / 18 val / 12 test over 185 tickers**
(83.8 / 9.7 / 6.5%).

### D6 — The model sees a bounded excerpt, not the full filing
Risk sections alone run 36K–146K characters. Excerpt layout is deliberate (`src/labels/excerpt.py`):
statements first (that is where the numeric answers live, and it keeps them intact if anything
downstream truncates), then narrative, with MD&A **windowed around the XBRL figures** rather than
truncated from the top.

### D7 — The teacher labels the excerpt, not the full filing
This one changed the design. Labeling the whole document while training on a truncated excerpt asks
the model to produce risks its input never contained — that is *training confident hallucination*,
and it would have stayed invisible until error analysis on Day 9.

### D8 — Ungrounded numeric labels are dropped to null, not shipped
`check_grounding()` asks whether each XBRL figure actually appears in the excerpt text. A label the
input does not support is unlearnable. `None` is a legitimate target — the schema already expects
null for concepts a filer does not report, and Truist genuinely files no total-revenue tag.

This doubles as the last line of defence against a mis-selected XBRL fact.

### D9 — Filings whose Item 1A is a cross-reference are flagged, not dropped
Wells Fargo and US Bancorp satisfy the disclosure requirement by pointing at their Annual Report,
leaving ~200 characters of cross-reference. An empty risk list is the *correct* output there, not a
miss. Their financial labels are still valid, so the filings stay in and only risk-factor scoring
excludes them (`src/labels/build.py:106`).

### D10 — XBRL facts are selected explicitly, not via the convenience API
`edgartools`' `Financials.get_financial_metrics()` is a fallback, never the primary source. See
[I2](#i2--the-convenience-dict-was-wrong-for-34-of-182-filings). Selection rules now in
`src/data/xbrl_facts.py`: an ordered concept-priority list per field, undimensioned facts only,
facts on the face of a primary statement over note disclosures, longest period wins (the annual
figure over the Q4 that ends the same day), and **no prior-year fallback** — a filing with no fact
for its own period is a data problem, not a reason to label last year's number as this year's.

---

## 3. Training setup

### D11 — Loss computed only on the assistant turn
~87% of each example is filing text the model is always given. Training on it wastes the run.
Mis-masking produces no error at all, which is why `tests/test_train_data.py` asserts on exact
token positions rather than shapes.

### D12 — Middle-out truncation
Statements sit at the head of the excerpt and risk factors at the tail; both carry labels. MD&A in
the middle carries none, so it is the only safe thing to drop. A target that does not fit raises
rather than being silently cut.

### D13 — QLoRA + Unsloth on a rented GPU, not local
Fastest single-GPU stack for an 8B model, and the laptop is disk-constrained (see
[I4](#i4--macos-purged-the-entire-project-directory)). Local venv stays at base + `labels,eval,dev`
(~500MB); `train` extras and model weights never land on this machine.

### D14 — A smoke path that runs without a GPU
`src/train/sft.py:35` swaps in an ungated tiny model with the same chat-template and tokenizer
machinery, small enough to train on CPU/MPS in under a minute. Catches loop bugs before any rented
GPU time is billed.

---

## 4. Evaluation

### D15 — The frontier baseline is Sonnet 5, deliberately **not** Haiku
Haiku 4.5 generated the risk-factor gold labels. Scoring Haiku here would measure self-agreement and
report a meaningless near-perfect F1 (`src/eval/predict.py:67`).

### D16 — The baseline is not handicapped
Sonnet 5 thinks by default, and thinking shares the `max_tokens` budget with the answer. Rather than
disable it, the budget is doubled. "Matched a frontier model" must not be a claim about a weakened
opponent. Cost of the choice is recorded per run.

### D17 — Two prompt modes, both published
- `trained` — the exact prose system prompt the student is fine-tuned on, no field names.
- `schema` — that prompt plus the JSON schema of the target.

The student does not need the schema in-prompt because 152 training examples put it in the weights.
Under the prose-only prompt Sonnet scores **0%** — it invents its own field names (`total_revenues`,
`diluted_eps`, risk factors as bare strings). That is a real measure of what fine-tuning teaches,
but it would be a strawman as *the* competitive baseline. The schema-prompt run is the number the
cost comparison rests on; both are reported.

Prompt mode is part of a run's identity, not a footnote — the prediction cache is keyed on it, and
the report names it.

### D18 — Macro (per-filing mean) is the headline F1
One verbose filer discloses 46 risks where another discloses 5. Micro-averaging would let that
single filing set the corpus number.

### D19 — `shares_outstanding_diluted` is unscored
Absent from most filings and not covered by the grounding check, so it injects noise rather than
signal.

### D20 — Schema validity is reported strictly and leniently
Strict = the raw string is JSON. Lenient = JSON survives after stripping markdown fences. Sonnet's
baseline is 100% lenient / 25% strict — a gap worth showing rather than averaging away.

---

## 5. Engineering guardrails

These exist because something broke first.

### D21 — A failed run must never clobber a good dataset
Split files were written unconditionally at the end, so one billing outage silently replaced
`train.jsonl` with an empty file. Now the run refuses to write a partial dataset unless
`--allow-partial` is passed.

### D22 — Non-retryable API errors abort the whole run immediately
A 400 for exhausted credit, a bad key, or an unknown model cannot succeed on retry. Each filing was
still burning three attempts with backoff — ~14s apiece, roughly **45 minutes of pointless waiting**
across the full ticker list. Now raises `TeacherUnavailableError` on first occurrence.

### D23 — Teacher labels are cached per filing
Reformatting examples never re-pays for labeling. This is why `refresh_facts.py` repairs facts
**in place** rather than re-running ingestion: section text stays byte-identical, the cache still
hits, and a full relabel after the XBRL fix cost $0.

### D24 — Ruff's rule set is pinned explicitly
Ruff's implicit defaults widen between releases. Unpinned, lint results depend on whichever version
a fresh venv happens to install — 57 errors appeared on untouched code after one such widening.

---

## Incident log

### I1 — Grounding was 14–57%, not the expected ~100%
**Symptom.** The first grounding report showed only 14–57% of numeric labels present in the model's
input.

**Cause.** `net_income`, `total_assets`, and `eps_diluted` appear *nowhere* in Items 1/1A/7. They
live only in the Item 8 financial statements, which ingestion never captured.

**Fix.** Added `src/data/statements.py`, re-ingested. All four fields went to **100%**.

**Why it mattered beyond the bug.** The fix made the task better rather than just bigger: filers
write "Net sales" vs "Total revenues" vs "Net revenue", so mapping company-specific line items onto
canonical fields is genuine normalization work — with XBRL as the answer key.

**Follow-on.** A unit test then caught a bug in the checker itself: `numeric_variants(5000)` emitted
`"5"`, which matches the "5" inside "2025" and reported ungrounded labels as grounded. Requiring ≥3
digits fixed it, and grounding still measured 100% — so the number is real, not a false positive.

### I2 — The convenience dict was wrong for 34 of 182 filings
`Financials.get_financial_metrics()` was silently supplying figures that disagreed with the filed
XBRL tags:

| Filer | Convenience dict | Filed tag |
|---|---|---|
| Wells Fargo (revenue) | 16.1B | 83.7B |
| American Tower (total assets) | 486M | 63.2B |
| Truist (revenue) | 471M | *no revenue concept filed* |

Systematically wrong for every bank and insurer in the corpus. **34 of 182 filings carried at least
one wrong figure.** Left alone, the fine-tuned model would have learned to produce them confidently.

Four distinct fact-selection defects surfaced during the fix:
- Prefix concept matching → added `exact=True` (ABBV net income 7M → 4,226M)
- Dimensional facts taken as headline → filter `is_dimensioned` (EA EPS −0.22 → 3.51)
- Q4 and FY sharing a `period_end` → sort by longest duration (ABBV 1,816M vs 4,226M)
- NaN labels — `float("nan")` parses fine and compares unequal to everything, so a NaN label scores
  neither right nor wrong. Added `_is_missing()`.

Grounding alone would not have caught American Tower: those digits happen to appear elsewhere in the
text. That is why the fix is at fact *selection*, with grounding as the second net.

### I3 — Teacher run against a credit-exhausted account
Surfaced [D21](#d21--a-failed-run-must-never-clobber-a-good-dataset) and
[D22](#d22--non-retryable-api-errors-abort-the-whole-run-immediately). Both are now covered by tests.

### I4 — macOS purged the entire project directory
**2026-07-26, ~11:46.** macOS's `deleted` daemon caught `VQ_LOWDISK` on the data volume and reclaimed
the whole `Finetuning-EDGAR/` tree — source, `.venv`, and `.git` — trying to free ~11GB. Neither user
nor tooling deleted anything; sibling projects in the same parent directory survived, and the tree
was simply the largest recently-written thing on the volume.

**Recovery.** Claude Code's file-history backups live *outside* the project at
`~/.claude/file-history/`, keyed by `backupFileName` in each session transcript's
`file-history-snapshot` records. **27 of 32 files restored byte-exact**; the 5 written after the last
snapshot were rebuilt by replaying their last `Write` payload plus every subsequent `Edit` in
timestamp order. Test count before the wipe was 78; after reconstruction, 78. That match is the
evidence the restore was faithful.

**Cost.** The teacher cache and `data/` are gone, so relabeling is no longer free (~$2, ~40 min).
Git history is flattened into one restore commit — the per-day commit trail is unrecoverable.

**Prevention.** Local venv stays lean, no model weights on this machine, `df -h /System/Volumes/Data`
before any large download. Repo now pushed to GitHub.

---

## Open questions

- **Llama 3.1 8B is gated** and `HF_TOKEN` is empty (401 on both repos). Either accept the license
  and supply a token, or switch to ungated `Qwen/Qwen2.5-7B-Instruct`. Must settle before renting a
  GPU.
- **Teacher-agreement rate is unmeasured.** ~50 hand-checked examples still owed; the project claims
  teacher labels are spot-checked and that number needs to exist.
- **Base-model baseline not yet run.** It is the "before" in the headline comparison and should be
  the first thing the rented GPU does.
