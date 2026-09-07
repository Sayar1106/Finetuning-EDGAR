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

### D30 — A100 80GB on RunPod, chosen for reliability rather than price
A pre-flight review of `src/train/` (2026-09-07) returned NO-GO on two counts: the paid run would
have been the first-ever execution of the training loop, and checkpoints had nowhere durable to
land. The second is fixed in code (see [D31](#d31--checkpoints-leave-the-box-as-they-are-written));
the first is a runbook step. It also flagged that this run had no cost estimate, unlike every other
spend in the project. This entry is that estimate.

Sizing, from the processed splits on disk: 186 training examples averaging ~8.2K tokens give
~1.5M tokens per epoch, ~4.6M over three epochs, across ~70 optimizer steps at an effective batch
of 8. The longest example is ~11.6K tokens, so `max_seq_len: 12288` clears it with headroom. These
are a chars/3.6 proxy; `python -m src.train.measure` replaces them with the real distribution on
the box, and is step 2 of the runbook for exactly that reason.

**The finding that decided it: the run costs roughly $2-4, not $40.** At Unsloth throughput on an
A100, 4.6M tokens is 30-45 minutes of compute — call it 60-90 minutes wall-clock including the 16GB
weight download, the smoke run, `measure`, and both eval passes. Against a <$40 project budget with
~$1.74 already spent ($1.40 labeling, $0.34 Sonnet eval), price is not the binding constraint. So
the instance was chosen to avoid repeating the run, not to shave a dollar off it:

- **`NVIDIA A100-SXM4-80GB` on RunPod secure cloud, $1.59/hr, on-demand.** Unsloth wheels are
  well-trodden on Ampere, which matters because `src/train/sft.py:66` falls back to transformers+peft
  on `ImportError` — a silent ~2x slowdown, not a failure. 80GB also buys
  `per_device_train_batch_size: 2` with `gradient_accumulation_steps: 4`: identical effective batch,
  meaningfully faster than the batch-1 config a 40GB card forces. Provision a 50-100GB volume for
  base weights plus checkpoints.

  SXM rather than the PCIe variant this entry first named: the live catalog (2026-09-07) reports
  `NVIDIA A100 80GB PCIe` at availability LOW against SXM's HIGH, at an identical $1.59 secure
  price, and SXM is the faster interconnect. The cheaper option was queued behind stock, not behind
  money — which is the same reasoning as everything below, applied to a fact the estimate could not
  have known. Community cloud is $1.39 vs $1.59 secure; $0.30 over the whole run does not buy out of
  a community host's reliability.
- **Not spot.** Hub checkpointing now makes pre-emption survivable, but at these totals spot saves
  about a dollar and costs an interruption.
- **Not a 24GB 4090** (~$0.30/hr, would probably fit): the savings are noise against an OOM
  discovered mid-run.
- **Not an H100** (~$2.50/hr): finishes sooner for about the same total, and adds version friction
  on a stack that has never been executed.
- **Not Colab**: session limits and no guaranteed A100 are wrong for a run worth not repeating.

One instance stays alive across the whole sequence — smoke, `measure`, base-model baseline,
fine-tune, fine-tuned eval, Q5 throughput. Re-downloading 16GB of weights is the only step here
that wastes real money.

**Actuals, 2026-09-07.** The session ran 73 minutes wall-clock for **$1.94**, inside the $2-4 estimate.
The token sizing was not as good: `measure` on the real tokenizer reports median 5,911 and max 8,515
tokens against the proxy's median ~7,950 and max ~11,557, so the corpus is **~35% smaller than
estimated** — 1,127,859 training tokens per epoch, not ~1.5M. Training itself took 20.3 minutes, and
zero examples truncated at `max_seq_len: 12288`, which the proxy could only guess at. The estimate's
*rate* and *instance* held; its *volume* was high. Worth keeping as a calibration note: a chars/token
proxy on JSON-heavy text ran a third over, in the direction that makes a budget look worse than it is.

### D31 — Checkpoints leave the box as they are written
Until the pre-flight, `src/train/sft.py` wrote checkpoints only to local `outputs/`, and the repo
had no `push_to_hub`, `hub_model_id`, or `resume_from_checkpoint` anywhere. On rented hardware a
dropped session or a reclaimed instance would have destroyed every epoch already paid for, with
nothing to restart from. A `hub` block in `configs/sft_llama31_8b.yaml` now drives
`TrainingArguments`.

`strategy: checkpoint` rather than `every_save`: the former uploads the resumable `last-checkpoint`
folder, LoRA weights plus optimizer state. Because the frozen 4-bit base is never uploaded, that is
hundreds of MB rather than tens of GB — cheap enough that resumability is worth paying for.
`--resume` is the consuming half; pushed checkpoints nothing can restart from would not justify the
upload.

Two smaller choices. The final push runs *after* `train_config.json` and `train_metrics.json` are
written, so the reproducibility files travel with the adapter instead of the weights alone. And an
enabled hub with an empty `HF_TOKEN` warns loudly and falls back to local disk rather than failing —
the same silent-failure class as the W&B gate, which is worth a log line rather than a discovery at
the end of a paid run. Smoke mode forces the whole block off; a throwaway 0.5B adapter has no
business creating a Hub repo.

The repo name `Llama-3.1-8B-edgar-10k-qlora` carries the prefix the community license requires of
derivatives ([Q9](#q9--licensing-of-the-published-artifact)), with the reason in a config comment so
it does not read as arbitrary later. It stays private until the model card is written: publishing is
a deliberate step, not a side effect of training.

### D32 — Three epochs were configured; epoch 2 is what shipped
The 2026-09-07 run trained the full three epochs in `configs/sft_llama31_8b.yaml`, but validation
loss turned:

| epoch | eval_loss |
| --- | --- |
| 1 | 0.4583 |
| 2 | **0.4459** |
| 3 | 0.4709 |

Train loss kept falling to 0.370 while eval loss rose in the third epoch — ordinary overfitting on
186 examples. Because `load_best_model_at_end: true` with `metric_for_best_model: eval_loss` is set,
the Trainer restored `checkpoint-48` before saving, so **the published adapter is the epoch-2 model,
not the epoch-3 model the epoch count implies.** Anyone reading `num_train_epochs: 3` and assuming
they know which weights are on the Hub would be wrong, which is the reason this is written down.

This is [Q6](#q6--model-selection-on-eval_loss-but-you-report-exact-match-and-f1) earning its keep
rather than a hypothetical: selecting on `eval_loss` is what prevented shipping the worse model. The
config is deliberately left at 3 — the third epoch is what produces the evidence that 2 is right, and
a future run on a larger corpus should re-check rather than inherit the answer.

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

The student does not need the schema in-prompt because 186 training examples put it in the weights.
Under the prose-only prompt Sonnet scores **0%** — it invents its own field names (`total_revenues`,
`diluted_eps`, risk factors as bare strings). That is a real measure of what fine-tuning teaches,
but it would be a strawman as *the* competitive baseline. The schema-prompt run is the number the
cost comparison rests on; both are reported.

Prompt mode is part of a run's identity, not a footnote — the prediction cache is keyed on it, and
the report names it.

**Measured 2026-09-07, and one half of this did not hold.** The student was evaluated under both
prompts. "Does not need the schema" is confirmed — under the prose-only prompt it scores 100% strict
schema validity and 97.9% numeric accuracy, where the base model scores 0% on both. But the schema is
not *inert* for the student either: with it, risk category F1 rises 77.3% -> 82.0% and category
accuracy among matched risks 72.4% -> 80.1%, while numeric accuracy falls 97.9% -> 93.8%. The
in-weights claim holds for structure and numerics; for risk categorization the schema still carries
information the 186 examples did not fully install. The headline row stays `trained` — it is the
prompt the student was fine-tuned on — but the gap is a real result, not noise to omit.

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

### D25 — Every headline metric carries a bootstrap interval
`score_dataset` computes 95% percentile-bootstrap CIs by default (10,000 resamples, seed 0) and
`format_report` prints them beside every point estimate. Reporting a bare rate at n=12 invites a
comparison the data cannot support, so the interval is on by default rather than opt-in.

**The resampling unit is the filing, not the field instance.** The four numeric fields within one
filing are correlated — a filer reporting in thousands tends to produce a scale error on all four at
once — so resampling instances would treat 48 correlated outcomes as 48 independent ones and report
an interval roughly √4 too narrow. `tests/test_eval.py` asserts the width is consistent with 12
clusters rather than 48 draws, which is the assertion that would catch a regression to instance-level
resampling.

Metrics no resample can evaluate are **omitted rather than zeroed** — risk F1 when every filing
satisfies Item 1A by cross-reference is undefined, and `0.0 [0, 0]` would read as a measured failure.

*Known limitation:* the percentile bootstrap pins to the observed extremes, so near 100% the upper
bound is 100% and the interval is one-sided in effect. Fine for the comparison being made; worth
naming rather than presenting the interval as symmetric.

### D26 — Teacher labels are audited by a blind human sample, not eyeballed
Every risk-factor number this project reports is scored against gold labels Claude Haiku wrote, so
each of them is conditional on the teacher being right. `src/labels/audit.py` measures that
conditional instead of asserting it: `sample` draws a review sheet, `review` collects verdicts, and
`score` reports agreement with intervals. The worksheet is checked in at `docs/audit/` — a
"spot-checked by hand" claim with no worksheet behind it is the thing the module exists to replace.

**Three verdicts per risk, not one**, because they fail differently and the fixes differ:
*grounded* (is the risk actually in the text the teacher was given?) catches hallucinated labels,
the failure that makes training data actively harmful rather than merely noisy; *category* is the
headline, because `category_f1` and `category_accuracy` are what the eval scores; *summary* catches
faithful-but-boilerplate prose that would fit any registrant.

**The reviewer picks the category before seeing the teacher's.** Showing a label and asking "do you
agree?" measures agreement with an anchor — a plausible-looking answer gets accepted that the
reviewer would never have produced unprompted. `review` withholds the teacher's category until the
human commits, then compares programmatically. Only the category verdict can be blinded this way,
which is a further reason it is the headline rather than the grounding rate.

**The sample is clustered, and so is the interval.** A uniform draw of 50 risks would land in ~50
distinct filings, each demanding its own read of Item 1A. So `sample` draws filings first (12), then
risks within them (4 each) — ~48 items from 12 readings. That makes the items non-independent, so
`score` resamples *filings* in the bootstrap, the same unit as [D25](#d25--every-headline-metric-carries-a-bootstrap-interval).
Below 3 reviewed filings the interval is withheld entirely: a one-cluster bootstrap redraws the same
filing every time and prints `[100.0%, 100.0%]` off four items, which is worse than no interval.

**Two strata, scored separately and never pooled.** The `random` stratum is uniform over the corpus,
so it estimates label quality across the dataset. But the test split is 12 of 216 filings, so a
12-filing uniform draw is expected to contain fewer than one — and the first draw contained zero.
Test gold labels are what every reported risk-factor number is scored against, so a `test` stratum
(3 filings) oversamples them deliberately. Pooling the two would bias the corpus estimate toward
whatever the test split happens to look like.

Two things the audit deliberately does *not* cover. Numeric labels are not audited: they come from
XBRL, are exact by construction, and are already checked against the excerpt by `check_grounding`.
And the missed-risk count yields an **upper bound** on recall, not a measurement — a human reading
Item 1A once catches omissions the teacher made loudly, not every risk buried in a subordinate
clause.

Verdicts are pinned to a digest of the labels they judged. Rebuilding the dataset changes the
sentence at a given index, and silently re-pointing an old verdict at a new label would turn the
worksheet into fiction; `score` excludes stale rows and says how many. Same reasoning as the teacher
cache key in [D23](#d23--teacher-labels-are-cached-per-filing).

### D27 — The published baseline is the run whose artifact is checked in
The frontier baseline was re-run on 2026-08-02 because its artifacts no longer existed: the numbers
in the README had been carried forward in prose while `data/eval/reports/` and the prediction cache
were gone, lost with [I4](#i4--macos-purged-the-entire-project-directory). A headline number that
cannot be traced to a file is an assertion, which is the same failure
[D26](#d26--teacher-labels-are-audited-by-a-blind-human-sample-not-eyeballed) exists to fix on the
labeling side.

The re-run superseded the earlier figures rather than being averaged with them, and the report JSON
is committed alongside. `.gitignore` already encodes this split — cached predictions are regenerable
and ignored, reports are results and are kept — so the rule is now uniform: **the published number is
whichever run left an artifact in the repo.** Averaging two runs would have produced a number no file
contains and no command reproduces.

That the figures moved at all is the finding, not a nuisance: it is a measurement of inference
variance the project previously had no evidence for, and it sets a floor on what margin can be
claimed later ([Q7](#q7--reproducibility-is-partial)). The cost figure moved too, and that one is
still unexplained — see Q5.

### D28 — The taxonomy ships to the teacher undefined, and the audit found it
[D4](#d4--closed-risk-factor-taxonomy) argues for a closed category set because it makes risk
categorization scoreable as multi-label F1 rather than fuzzy string matching. That argument holds.
What it never did was say what the twelve categories *mean*. `SYSTEM_PROMPT` in
`src/labels/teacher.py` interpolates the bare tuple — "Use only these categories: market,
operational, regulatory, …" — with no gloss, no examples, and a single tie-break: choose the
category the filing emphasizes most. `src/labels/schema.py` is the same twelve strings.

So the teacher was never given a definition; it invented boundaries from the label names, and the
hand audit ([D26](#d26--teacher-labels-are-audited-by-a-blind-human-sample-not-eyeballed)) is what
surfaced them. Of the first 20 reviewed risks, 6 category disagreements were **all** adjacent-label
boundary calls — `legal`/`regulatory`, `regulatory`/`operational`, `supply_chain`/`operational`,
`competitive`/`operational`, `cyber`/`operational`, `competitive`/`market` — and none was a
substantive misreading of the risk. `operational` is one side of four of the six.

The `cyber` boundary is the clearest case, because the teacher's implicit rule is recoverable from
usage. Across the 2,834 labeled risks, `cyber` is 108 of them, and it tracks **adversary versus no
adversary**: breach, attack, unauthorized access and data privacy are `cyber`; reliability, outage,
internal control and integration failure are `operational`. Of 63 risks mentioning information
systems or data integrity, 40 are `cyber` and 19 `operational`, split along that line. UnitedHealth's
filing shows the teacher applying it deliberately — "Data integrity and information systems
reliability" is `operational`, while "Cyber-attacks and data security breaches" in the same filing is
`cyber`.

That rule is defensible and internally consistent. It is also nowhere in the prompt, so a human
reviewer reading "cyber" in its plain sense — anything about information systems — disagrees with
the teacher while both parties are being reasonable. The disagreement measures an undefined
taxonomy, not a wrong teacher.

The glosses recoverable from usage are written down in [`taxonomy.md`](taxonomy.md), along with the
three themes where they are **not** recoverable: pandemic/health crisis splits four ways across
`operational`/`market`/`supply_chain`/`macroeconomic` with a 38% mode, tariffs the same at 38%, and
geopolitical conflict at 43%. Those are not boundaries a reviewer could guess, and a human/teacher
disagreement on them carries no information — the audit flags them rather than scoring either answer
as correct. They are also the cases that need an authored rule rather than a documented one.

**Consequence.** The headline agreement rate is partly a measure of prompt underspecification. A
one-line gloss per category would likely absorb most of these disagreements, but changing the prompt
invalidates the existing labels for all 220 filings and re-pays the teacher run, so it is not being
done inside v1. What is done instead: the audit records the boundary pairs, and the reported
agreement rate is qualified by them rather than presented as a clean measure of teacher correctness.

### D29 — The audit reports blind agreement, not the recorded rate
The audit is complete: 60 of 60 sampled risks reviewed across 15 filings. `score` reports **89.6%**
category agreement on the random stratum (95% CI [79.2%, 97.9%], 12 filings) and **66.7%** on the
test stratum (95% CI [25.0%, 100.0%], 3 filings). Grounding and summary faithfulness are 100% in
both. Those are the numbers the tool prints, and they are not the numbers this project publishes.

**Eight of the agreements were manufactured after the fact.** `review` withholds the teacher's
category until the human commits, which makes each verdict blind. But it prints `DIFFERS` immediately
afterward, and on eight risks the reviewer re-read the text and changed their answer to match. Every
such edit is recorded in the row's `note` with the original verdict preserved, so both rates stay
computable:

| Slice | n | Recorded | Blind | Blind 95% CI |
|---|---:|---:|---:|:---:|
| random stratum | 46 | 91.3% | **76.1%** | [65.2%, 87.0%] |
| test stratum | 12 | 66.7% | **58.3%** | [25.0%, 100.0%] |
| All scorable | 58 | 86% | **72%** | — (strata are not pooled) |

The 14-point spread is not a reviewer failing; it is the predictable shape of revising under
feedback. The revision is **one-directional** — a verdict that happened to agree with the teacher is
never re-examined, because nothing prompts a second look. So errors that coincide with the teacher
survive and errors that diverge get corrected, and the recorded rate is inflated by construction.
Several of the eight had genuine substance (PSA #7 turned on the word "liability" in the title; DG #4
on a data-security clause the first read missed), which is exactly why the effect is hard to see from
the inside: each individual revision looks like a correction. **The blind rate is the one that
answers D26's question**, and it is what the README will carry, with the recorded rate alongside it
as the measure of how much seeing the answer moved the reviewer. One further edit — MET #4 — was a
keying error rather than a revision, tagged `corrected`; it restores the intended blind verdict and
scores as a blind match.

**`score` now computes both rates itself.** It previously read `human_category` as recorded — no
notion of a blind rate — and ignored the `UNINFORMATIVE` flags, keeping both MET rows in the
denominator (89.6% over 48 against the exclusion-aware 76.1% over 46). Both are fixed:
`blind_category` recovers the pre-revision verdict, flagged rows drop out of the category rate while
still counting toward grounding and summary, and the report prints the blind figure as the headline
with the recorded one beside it. The rates in this document are now `score` output rather than hand
arithmetic, and the confusion table lists blind disagreements to match.

The three things that change a row's score — `revised`, `corrected`, `UNINFORMATIVE` — live in the
free-text `note` field, which makes them a parsing target. They are documented as a small grammar at
the top of `audit.py`, and `check_notes` runs before every `score` to fail loudly on a marker that
does not parse: a `revised` note with no recoverable verdict, a blind category that is not in the
taxonomy, the word UNINFORMATIVE appearing outside a real marker. A silently misparsed note would
move a published rate, which is precisely the failure this audit exists to prevent.

**The gap survives resampling.** The bootstrap now draws the paired difference (recorded − blind)
within each resample, since both rates come from the same rows. On the random stratum the gap is
**+15.2%, 95% CI [4.5%, 27.1%]** — the interval excludes zero, so anchoring is established on this
sample rather than merely observed. On the test stratum it is +8.3%, CI [0.0%, 25.0%], which does
not exclude zero; with one revision across three filings it could not have.

**The taxonomy blocks did not produce a clean comparison.** The first 20 risks were reviewed with no
written definitions and the remaining 38 with `taxonomy.md` open, which was intended to measure
whether the glosses help. Blind agreement barely moved (70% → 74%) while *recorded* agreement fell
from 100% to 79% — an artifact of block 1 having revised away every one of its disagreements. The
blocks also differ by stratum and by company, so they are not comparable, and no claim is made from
them. Whether written glosses raise agreement remains untested.

**Two findings worth carrying forward.** The test stratum is the weaker one on every axis: 58% blind
agreement against the random stratum's 76%, and implied teacher recall of **81.5%** (5 risks missed
of 22) against 96.0% (6 of 145). That is the split the headline eval scores against, so the gold
labels behind the README's risk-factor F1 are the least reliable in the corpus. The caveat is that
three filings is at the floor of what `MIN_CLUSTERS_FOR_CI` permits — the [25.0%, 100.0%] interval
spans nearly the whole range, and the gap between strata is not established by this sample. It is a
reason to widen the test-stratum audit before trusting the comparison, not a result.

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

### I5 — Three tickers in the universe stopped resolving
**2026-07-29.** The re-ingestion logged `Company not found` for `MMC`, `ANSS`, and `NVEE` — 3 of 185.
Not a bug in the fetcher: EDGAR's `company_tickers.json` lists only a company's *current* symbol, so
a symbol that stops being current is simply absent. Two distinct causes, and the difference matters
because only one of them means the company is still filing:

| Ticker | Company | Cause | Still files? | Latest 10-K |
|---|---|---|---|---|
| `MMC` | Marsh & McLennan | renamed, now trades as **MRSH** | yes | FY2025 |
| `ANSS` | Ansys | acquired by Synopsys, delisted, `tickers: []` | no | FY2024 (final) |
| `NVEE` | NV5 Global | acquired by Acuity, delisted, `tickers: []` | no | FY2024 (final) |

The delisted two are kept rather than dropped: a delisted company's final 10-K is still a real filing
with real XBRL, so it is valid training data, and FY2024 is still past the ~Dec-2023 pretraining
cutoff ([Q3](#q3--how-do-you-know-the-base-model-hadnt-already-memorized-these-filings)). Their
presence is arguably a small plus — an acquired mid-cap is exactly the kind of filer whose text a
frontier model is least likely to have memorized.

**Fix.** `CIK_OVERRIDES` in `src/data/companies.py` pins each to its CIK (`62709`, `1013462`,
`1532961`) and `get_latest_10ks` prefers it over the symbol. A CIK is permanent through both a rename
and a delisting; a ticker is not. Every CIK was resolved by company name against SEC's map and
confirmed to have 10-K filings, not recalled from memory. All three verified live — FY2025, FY2024,
FY2024 respectively, all four numeric fields present, no parse warnings on any of them.

edgartools' own suggestion for `MMC` was `MMCP` (Mag Mile Capital), an unrelated microcap. Accepting
a fuzzy ticker match would have ingested the wrong company's 10-K and scored a model against it.

**The key stays `MMC`, not `MRSH`.** `src/labels/splits.py` assigns companies to train/val/test by
hashing the ticker, so renaming the key would move this company to a different split and invalidate
comparisons against every earlier run. A test asserts each override names a ticker still in `TICKERS`
and carries an `int` CIK — a string would be passed through as a ticker and fail the same obscure way.

*Generalization declined:* no automatic name-search fallback on `CompanyNotFoundError`. That is
exactly the mechanism that would have picked `MMCP`. Renames are rare enough to pin explicitly and
review in a diff.

---

## Anticipated questions

Known soft spots, with the answer written down before someone asks. A gap that is stated and
measured is a different thing from one that is missing.

### Q1 — "If XBRL already gives you exact numbers for free, why do you need a model at all?"
The premise question, and the honest answer is that XBRL is the *answer key*, not a substitute for
the task.

XBRL exists only for SEC registrants filing since ~2009 in a structured taxonomy. The model reads
**text**, which is what is available for private-company filings, credit agreements, foreign
issuers, pre-XBRL archives, and PDFs generally. And half the extraction schema — risk-factor
categories and summaries — has no XBRL counterpart at all ([D3](#d3--teacher-model-only-for-the-qualitative-half)).

The project's design deliberately uses the one domain where an exact answer key happens to exist, so
the extractor's accuracy can be measured against filed truth instead of against another model's
opinion. That is a statement about **eval methodology**, not about the deployment target.

### Q2 — "What are your error bars?"
Reported — see [D25](#d25--every-headline-metric-carries-a-bootstrap-interval). Every headline metric
carries a 95% percentile-bootstrap interval, and the eval report prints them next to the point
estimate.

The size of them is the thing to internalize: 12 held-out companies × 4 scored numeric fields ≈ 48
comparisons, so one field is 2.1 points and one filing is 8.3 points of schema-validity. The
measured baseline sits at 97.9% numeric accuracy with an interval of **[93.8%, 100%]**. A
fine-tuned-vs-baseline gap smaller than ~10 points on numerics is not distinguishable from sampling
noise at this test-set size. Close results are ties, and the table now says so.

Two independent reasons back that, and they compound. Sampling noise is what the bootstrap
estimates; on top of it sits generation noise, since the frontier baseline has no seed to pin and
re-running the same commit moved every headline figure
([Q7](#q7--reproducibility-is-partial)). An interval on a single run understates the spread of the
thing a reader actually cares about — where the number would land if the whole evaluation were
repeated.

The test set is 12 companies because splits are company-level over a 185-ticker universe
([D5](#d5--split-by-company-never-by-filing)); widening it means growing the universe, not
re-slicing.

### Q3 — "How do you know the base model hadn't already memorized these filings?"
Not yet controlled for. Llama 3.1's pretraining cutoff is ~Dec 2023 and 10-K filings are public web
text, so any test filing predating the cutoff is plausibly in the base model's training data. That
inflates the *base* baseline, deflates the measured improvement from fine-tuning, and muddies both
directions of the headline comparison.

Mitigation is cheap because ingestion takes the **latest** 10-K per company
(`src/data/pipeline.py`), and as of the 2026-07-29 re-ingestion this is now **measured, not assumed**:

| Split | Filings | Companies | FY ≥ 2024 | Fiscal years |
|---|---|---|---|---|
| train | 186 | 155 | 91.4% | FY2023:16 FY2024:17 FY2025:136 FY2026:17 |
| val | 22 | 18 | 90.9% | FY2023:2 FY2024:2 FY2025:15 FY2026:3 |
| **test** | **12** | **12** | **100%** | **FY2025:10 FY2026:2** |

**Every test filing postdates the cutoff by at least a year** — no test company's latest 10-K is
FY2023 or earlier, so the headline metric is not measurable on any filing the base model could have
memorized. The 18 FY2023 filings sit entirely in train and val, where memorization would if anything
work *against* the fine-tune's apparent gain rather than inflate it.

This is a property of the corpus, not a filter applied to flatter the result: the split is assigned by
hashing the ticker (`splits.py:23`) before fiscal year is known, so no filing was moved to make this
table look better. Reproduce with the ticker hash and `fiscal_year` from `data/raw/`.

### Q4 — "186 training examples? Why so few?"
One 10-K per company over a 185-ticker universe, with 18 companies carrying extra fiscal years from
an earlier partial run. As built on 2026-07-29: **220 filings from 185 companies → 186 train / 22 val
/ 12 test**, all 220 labeled with zero teacher failures. The original plan targeted 1,500–3,000.

Because splits are company-level, pulling *N* years per company is leakage-safe by construction —
`--n-filings 5` is a single flag and gives ~900 train examples with no change to the split logic.
That has not been done, for a reason worth stating rather than hiding: consecutive filings from the
same company are near-duplicate boilerplate, so 5× the filings is well short of 5× the effective
data. The cost objection is now **measured and weaker than it looked**: the full 220-filing labeling
run cost **$1.40**, so 5× is ~$7, not the ~$10 previously guessed from a rougher estimate.

That makes the honest constraint *marginal value*, not money. The right framing is an **ablation** —
186 vs ~900 — which is the data-scaling row the plan already wanted for Days 9–10. It converts "why
so little data?" into a measured curve, and at $7 the experiment is cheap enough that declining to
run it needs a better reason than cost.

### Q5 — "You claim ~1/20th the inference cost. Show the math."
**Still an estimate, and labeled as one** — but the measurable half is now measured, and the
unmeasured half has its arithmetic pre-specified so the gap is a missing number rather than a missing
method.

**What is measured.** `src/labels/teacher.py` records `response.usage` per call and
`src/labels/build.py` prints the run's spend, so dataset cost is a reported figure instead of a
reconstruction. The 2026-07-29 run: **220 filings, ~0.56 MTok in / ~0.17 MTok out on Haiku 4.5 →
$1.40.** Output averages 59.5 tokens per risk factor and input 2,526 tokens per filing (the 12K-char
`RISK_CHARS` cap does the work). The same corpus through Sonnet 5 would have been ~$4.19 — the
teacher-model choice in [D3](#d3--teacher-model-only-for-the-qualitative-half) is a 3× saving, which
is the kind of claim that should come with the number attached.

*Caveat on that $1.40:* per-token rates are hard-coded in `PRICING_USD_PER_MTOK` as published on
2026-07-29, and the figure above was computed from sampled token counts before the accounting shipped
— so it excludes the structured-output schema tokens on each call (~$0.07) and the two smoke-test
filings. The billing console is authoritative; the next labeling run will report its own total
directly.

**What is not measured, and exactly what would settle it.** The serving claim compares two things
priced in different units — API tokens versus GPU-hours — which `src/eval/run.py:97` already flags.
The comparison needs three numbers, none of which exist yet:

| Needed | How to get it |
|---|---|
| Throughput: tokens/sec for the 8B at 12K sequence length | Measure during the GPU rental, on the same 12 test filings |
| GPU cost: $/hour for the named instance | The rental invoice |
| Tokens per filing: input + output at inference | Already known — ~24.7 KB excerpt ≈ 6.2K input tokens, ~1K output |

Cost per filing is then `(input + output tokens) / (tokens/sec) / 3600 × $/hour`, set against the
frontier baseline's measured **$0.34 over 12 filings ($0.028/filing)** — the 2026-08-02 run,
104,311 input / 13,193 output tokens at the checked-in $2/$10 per Mtok. Until those three exist the
README says "estimate," and any ratio quoted — 1/20th or otherwise — is a projection.

This figure was previously recorded as $0.71 ($0.059/filing). The re-run halved it against the same
price table, so the earlier number reflects roughly twice the tokens — most likely both prompt modes
summed rather than the schema run alone ([D17](#d17--two-prompt-modes-both-published)). That is a
hypothesis, not a finding: the original run's artifacts were lost and the `trained`-mode run has not
been repeated to confirm it.

### Measured 2026-09-07 — and the ~1/20th claim does not survive
The three missing numbers now exist, from the fine-tuned `trained`-mode run on the 12 test filings
(`docs/runs/fteval.log`, `data/eval/reports/ft-llama-trained_trained_test.json`):

| Needed | Measured |
| --- | --- |
| Throughput | 696s of generation for 12 filings — **58.0s/filing, ~99 tok/s end to end** |
| GPU cost | **$1.59/hr**, A100-SXM4-80GB secure ([D30](#d30--a100-80gb-on-runpod-chosen-for-reliability-rather-than-price)) |
| Tokens per filing | 4,912 in / 805 out (58,944 + 9,664 over 12) |

Running the pre-specified arithmetic: `696s / 3600 × $1.59 = $0.307` for 12 filings, or
**$0.0256/filing against Sonnet's measured $0.0283 — a ratio of 0.91×.**

**That is parity, not a twentieth**, and it is the generous reading: it counts only generation, and
excludes the ~13 minutes of model loading the API never pays for. Reaching 1/20th at this rate would
take **18.1× more throughput**, or the same throughput at $0.09/hr. Neither is available by tuning a
flag — it needs batched or continuous-batching inference (vLLM, TGI) serving many filings per GPU-second,
which this project has not built and therefore cannot claim.

So the caveat this section always carried turned out to be the finding, not a hedge: a rented GPU
billed by the hour is only cheap at high utilization, and at 12 filings sequentially decoded the
frontier API wins. **The supportable claim is "matches Sonnet's content accuracy at parity cost, with
a measured path to lower cost through batching" — and the 1/20th figure should not be quoted until
something measures it.** Where the student does win outright is strict schema validity, 100% against
Sonnet's 33.3%, which is an output-discipline result rather than a cost one.

### Q6 — "Model selection on `eval_loss`, but you report exact-match and F1?"
Yes — `metric_for_best_model: eval_loss` with `load_best_model_at_end`. Cross-entropy on held-out
filings is a proxy for the metrics that are actually reported. A `compute_metrics` callback scoring
schema-validity and numeric match per epoch would select on the real objective; it was traded away to
keep rented-GPU time down at 3 epochs, where checkpoint choice is a weak lever. Worth revisiting if
the run is extended.

### Q7 — Reproducibility is partial
Seeds are set (`training.seed: 42`, propagated to LoRA init) and splits are a deterministic ticker
hash, so data assignment is stable. `pyproject.toml` pins only lower bounds (`>=`), so the same commit
resolves to different dependency versions over time — which already bit the project once through ruff
([D24](#d24--ruffs-rule-set-is-pinned-explicitly)).

**Half-closed as of 2026-07-29.** `requirements.lock` pins all 69 resolved packages for the base +
`labels` + `eval` + `dev` path — the environment that produced `data/processed/*.jsonl` and runs the
eval harness. The `train` extras are deliberately **not** locked: torch and friends are never
installed on this machine ([D13](#d13--qlora--unsloth-on-a-rented-gpu-not-local),
[I4](#i4--macos-purged-the-entire-project-directory)), so a lock generated here would be fabricated
rather than resolved. It gets generated on the GPU host at training time and committed as
`requirements-train.lock`.

So the accurate claim is: **the data and eval path is reproducible from a commit; the training path is
not yet.** That is narrower than `configs/sft_llama31_8b.yaml`'s comment, which should be softened to
match rather than left to overstate.

Two limits remain, and neither is a lockfile problem. Results come from a **single seed and a single
run**, so a margin between configurations carries no variance estimate — the bootstrap intervals in
[D25](#d25--every-headline-metric-carries-a-bootstrap-interval) quantify test-set sampling noise, not
run-to-run training variance, and those are different sources of error. And the dataset depends on
EDGAR, which is a live service: [I5](#i5--three-tickers-in-the-universe-stopped-resolving) is a
worked example of the same ticker list resolving differently four months apart.

**The frontier baseline is not reproducible run-to-run, and now there is evidence rather than a
caveat.** `src/eval/predict.py` sets no `temperature` — current models reject a non-default value —
so there is no seed to pin on the API side. Re-running the identical commit on 2026-08-02 moved
every headline figure: strict schema-validity 25% → 33.3%, numeric accuracy 95.8% → 97.9%, risk
category F1 76.7% → 79.2%. Each gap is one unit of the underlying count (one filing of twelve, one
field of forty-eight), and every superseded value falls inside the new run's interval. Two readings
follow, and both matter. The reassuring one: the bootstrap intervals of
[D25](#d25--every-headline-metric-carries-a-bootstrap-interval) are wide enough to cover the noise
they were built to represent. The disciplining one: at n=12 a single filing is 8.3 percentage points
of schema-validity, so **any fine-tuned-vs-frontier margin narrower than one filing is not a result**,
regardless of which side it favours. The published numbers are one draw, not the model's true score.

**The local model is not bit-reproducible either, which was a surprise.** Greedy decoding
(`do_sample=False`) is deterministic given identical inputs and identical arithmetic — but the
arithmetic is not identical across hardware. Re-running all three Llama configurations on 2026-09-07
on an **A40 with transformers 5.16.1**, against the A100/5.5.0 run of the same day
(`docs/runs/rerun-a40.log`):

| config | metric | A100, tf 5.5.0 | A40, tf 5.16.1 |
| --- | --- | ---: | ---: |
| base, schema | strict-valid | 0.0% | 8.3% |
| base, schema | risk match F1 | 40.9 | 45.6 |
| fine-tuned | numeric accuracy | 97.9% | 100.0% |
| fine-tuned | risk category F1 | 77.3 | 80.0 |

Same adapter, same prompts, same greedy decoding, same twelve filings. Different kernels round
differently, one argmax flips, and the generation diverges from that token on. Each gap is again one
unit of the underlying count. **The published figures are the lower of the two runs in every case
that moved**, which is the right direction for a number that goes in a README, and the rerun is
recorded here so the choice is visible rather than silent. The practical rule is unchanged and now
applies to the student as well as the baseline: a margin narrower than one filing is not a result.

### Q8 — Risk-factor F1 rests on lexical overlap
`src/eval/metrics.py:218` matches predicted to gold risks by title-weighted lexical overlap —
deterministic, parameter-free, and documented as a **lower bound**. A semantically correct
paraphrase that shares few tokens scores as a miss. Embedding similarity would be the obvious
alternative and was rejected for v1 because it introduces a model into the metric, which is the
thing this project's eval design exists to avoid. The conservative direction is the safe one for a
headline claim, but the number understates true performance.

### Q9 — Licensing of the published artifact
Llama 3.1's community license imposes conditions on derivatives — relevant because the plan
publishes weights and a model card to HF Hub. Qwen2.5-7B-Instruct is Apache 2.0 and carries none of
that, which was an argument for Qwen beyond merely dodging the gated-repo problem
([D30](#d30--a100-80gb-on-runpod-chosen-for-reliability-rather-than-price) settled it for Llama).
SEC filing content is public domain; the derived dataset is not encumbered.

**The obligations, read off the Agreement rather than recalled.** An earlier version of this entry
named two conditions from memory. The text was fetched 2026-09-07 from the canonical
`LICENSE` in `meta-llama/Llama-3.1-8B-Instruct` and §1(b) actually imposes five, of which the notice
string is exact and therefore the one worth getting character-perfect:

| § | Obligation | Where it is satisfied |
| --- | --- | --- |
| 1(b)(i)(A) | Provide a copy of the Agreement with the Materials or derivative | Distributed alongside the weights on the Hub |
| 1(b)(i)(B) | Prominently display "Built with Llama" on a related website, UI, blogpost, about page, or product documentation | `README.md` License section; `docs/model_card.md` |
| 1(b)(i) | Include "Llama" at the beginning of the derivative model's name | Repo name `Llama-3.1-8B-edgar-10k-qlora` |
| 1(b)(iii) | Retain the attribution notice verbatim in a "Notice" text file | [`NOTICE`](../NOTICE), byte-checked against the Agreement |
| 1(b)(iv) | Comply with the Acceptable Use Policy, incorporated by reference | Linked from the README and the model card |

The required string is `"Llama 3.1 is licensed under the Llama 3.1 Community License, Copyright ©
Meta Platforms, Inc. All Rights Reserved."` — reproduced in `NOTICE` and verified equal to the
Agreement's text after whitespace normalisation, because an inexact reproduction fails the very
requirement it exists to satisfy.

A reviewing agent flagged the earlier two-item list as unsourced, which it was. The lesson generalises
past this entry: a licence obligation is the last thing to take from a model's memory.

---

## Open questions

- **Base model: settled on Llama 3.1 8B Instruct.** Access to the gated repo was granted 2026-08-28
  and `HF_TOKEN` is set, so `config.json`, the tokenizer, and the weight shards all resolve 200 for
  `meta-llama/Llama-3.1-8B-Instruct` and for the base `Llama-3.1-8B`. `configs/sft_llama31_8b.yaml`
  already targeted it, so no config changed. The ungated Apache-2.0 fallback
  `Qwen/Qwen2.5-7B-Instruct` was not needed. What the choice costs is
  [Q9](#q9--licensing-of-the-published-artifact): the community license binds the derivative, so the
  published adapter needs a `Llama-` name prefix and a "Built with Llama" attribution in its model
  card. Qwen would have carried neither — that is the price of the stronger base, paid at publish
  time rather than train time.
- **Teacher-agreement rate: measured.** All 60 sampled risks across 15 filings are reviewed
  ([D26](#d26--teacher-labels-are-audited-by-a-blind-human-sample-not-eyeballed)). Blind agreement is
  72% overall — 76% random stratum, 58% test stratum — against a recorded 86%; see
  [D29](#d29--the-audit-reports-blind-agreement-not-the-recorded-rate) for why the two differ and
  which one is published. Grounding and summary faithfulness are 100% in both strata. `score` reports the
  blind rate directly, excludes the flagged rows, and validates the note markers it depends on, so
  the README's figures are tool output rather than hand arithmetic. What *is* checked automatically:
  every example with an empty gold risk list is flagged `risk_factors_by_reference` (4 of 220 — USB
  ×3, WFC), so none is silently scored against an empty target.
- **Base-model baseline: run 2026-09-07 and closed.** Both prompt modes are in
  `data/eval/reports/base-llama-{schema,trained}_*_test.json` (`docs/runs/baseline.log`). Under the
  trained-mode prose prompt the base model scores 0 on everything; under the schema prompt it reaches
  91.7% lenient / 0% strict schema validity, 83.3 numeric, 43.5 risk F1. The "before" in the headline
  comparison is a measurement now, not a placeholder.
- **Revenue is unrecoverable for 4 of 220 filings** (Duke Energy FY2025, NextEra FY2025, Truist
  FY2023 and FY2024); 11 training rows carry a null revenue once the grounding check has also run.
  Related to [I2](#i2--the-convenience-dict-was-wrong-for-34-of-182-filings): for these four the
  convenience dict has nothing either, so no warning fires. This does *not* corrupt labels — a field
  with no gold value is `NOT_SCORED` and drops out of both numerator and denominator
  (`metrics.py:391`). **The scored set loses nothing at all**: `test.jsonl` contains zero revenue
  nulls and every eval report shows `not_scored: 0` for revenue across 12 of 12 filings. All 11
  nulls are in train.

  Two things this entry previously said are wrong, corrected 2026-09-07 after a review checked them
  against the corpus. It is *not* true that "utilities and banks do not tag `us-gaap:Revenues`" —
  EXC ($24.3B), SO ($29.6B) and WFC ($83.7B) all resolve through the existing concept list with no
  fallback warning, so the sector framing was never the mechanism. And the fix it recommended is
  dead: `RevenueFromContractWithCustomerExcludingAssessedTax` has been implemented at
  `src/data/xbrl_facts.py:49` all along, while `InterestAndDividendIncomeOperating` resolves **0 of
  the 11** and would convert three defensible Truist nulls into grounded-but-wrong labels — it
  returns gross interest income, roughly twice the figure the gold convention takes. WFC settles it
  on a filing already in the test set: it prints both `Total interest income: 87,314,000,000` and
  `Total revenue: 83,699,000,000`, and the gold takes total revenue. **A null you can explain beats
  a confident wrong number**, so this stays a known limitation rather than a task. Duke and NextEra
  are combined multi-registrant filers whose consolidated facts carry an entity dimension and are
  dropped by the `is_dimensioned` filter (`src/data/xbrl_facts.py:136`) — no concept addition
  reaches them.
