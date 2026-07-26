# Future work

Ideas deliberately scoped out of v1, kept here so the reasoning survives. v1 is 10-K structured
extraction with XBRL as programmatic ground truth (see README).

---

## v2 candidate: fund strategy-compliance checking

**The idea.** A mutual fund / ETF prospectus states an investment policy in prose. An N-PORT filing
reports what the fund actually held. Extract the stated policy with an LLM, then verify it against
the real holdings — and flag drift.

ARKK's summary prospectus, for example, states:

> The Fund is an actively-managed exchange-traded fund ("ETF") that will invest under normal
> circumstances primarily (at least 80% of its assets) in domestic and foreign equity securities of
> companies that are relevant to the Fund's investment theme of next generation technology.

That is a machine-checkable claim: `pct_value` by `asset_category` / `investment_country` in the
fund's N-PORT holdings table answers it directly.

**Why it's a stronger project than plain fund-strategy extraction.** It is the only framing found
where the task both *has* objective ground truth and *needs* an LLM:

| Target | Has ground truth? | Needs an LLM? |
| --- | --- | --- |
| Fund holdings / fees / flows | Yes — N-PORT XML, exact | No — already structured |
| Fund strategy prose | No | Yes |
| **Stated policy → verified against holdings** | **Yes, derived** | **Yes** |

Extracting fund strategy prose *on its own* was rejected for v1 precisely because it inverts the v1
setup: the fund data that has ground truth doesn't need a model, and the part that needs a model has
no ground truth. That would have meant teacher-LLM labels judged by another LLM — the weakest eval
story available, and eval rigor is the point of this project.

---

### Spike findings (2026-07-25)

Roughly 35 minutes of live probing against EDGAR. Recorded so this doesn't need re-running.

**`edgartools` fund support is real and good.** `edgar/funds/` is ~9,000 lines: `Fund`, `FundSeries`,
`FundClass`, `Prospectus497K`, `FundReport` (N-PORT), `FundShareholderReport` (N-CSR), `FundCensus`,
`MoneyMarketFund`. Ticker resolution worked on every fund tried (VFIAX, FCNTX, PIMIX, ARKK).

**497K summary prospectuses parse well** — 17 of 20 sampled filings were usable: 10–45K chars, clean
"Principal Investment Strategies" text, structured per-share-class fee tables. The 3 failures were
supplements / sticker amendments filed under the same form type, which return boilerplate for
`investment_objective` and an empty `fees` frame. **Filter these out by length and by a populated
`fees` table**, not by form type alone.

**N-PORT holdings are excellent ground truth** — `FundReport.investment_data()` returns 26 columns
including `name`, `ticker`, `cusip`, `value_usd`, `pct_value`, `asset_category`, `issuer_category`,
`investment_country`, `is_derivative`, `derivative_type`, `notional_amount`, `counterparty`. Plus
`fund_info` for net assets, monthly flows, and per-class returns.

### Blocker to solve first: series alignment

Fund filings are made at the **trust** level and cover many sibling series, so there is a three-level
entity hierarchy (company → series → class) where 10-Ks have a flat CIK. `edgartools` does not
reliably resolve this, and **it fails silently** — returning a plausible-looking DataFrame for the
wrong fund rather than raising.

`Fund("ARKK").get_portfolio()` returns the holdings of the **ARK Israel Innovative Technology ETF**.

Checking the 3 most recent `NPORT-P` filings per fund against the requested series ID:

```
ARKK   want=S000042977  got=S000052299, S000071318, S000042978   0/3 match
VFIAX  want=S000002839  got=S000002839, S000002848, S000002847   1/3 match
FCNTX  want=S000006037  got=S000039220, S000006037, S000057289   1/3 match
```

Any label built on this without an explicit series check is silently mislabeled. This is a worse
failure mode than the XBRL unit/scale mismatches in v1, because it is wrong-but-believable.
`edgar/funds/series_resolution.py` exists and is the place to start; a hard assertion that
`FundReport.series_id == requested_series_id` is the minimum guardrail.

### Also known

- **Full prospectuses (485BPOS) are not directly usable.** 863K–2M chars; `filing.obj()` returns a
  raw `XBRL` object with no section splitting; one document covers 12–15 funds; heading wording
  varies (FCNTX returned zero occurrences of "principal risks"). Use 497K summary prospectuses
  instead, or budget real work for per-fund section splitting.
- Cost to pivot v1 mid-flight was ~2–3 of 14 days plus the unsolved alignment problem, which is why
  it was deferred rather than adopted.

### If picking this up

1. Solve series alignment first and prove it — that gates everything else.
2. Build the policy schema: `{claim_type, threshold, asset_class, geography, theme}` from prospectus
   prose (e.g. `>=80%` / `equity` / `domestic+foreign` / `next generation technology`).
3. Derive the check from N-PORT: aggregate `pct_value` by category, compare against threshold.
4. Objective, quantitative claims (percentages, asset classes, geographies) verify cleanly. Thematic
   claims ("relevant to next generation technology") do not — either scope them out or treat them as
   a separate judged task, and be explicit about which is which.

---

## Cheaper alternative kept in scope for v1

If the strategy angle is wanted without the pivot: companies describe strategy in Item 1 (Business),
which the v1 pipeline already ingests. Adding a `strategy_summary` field to the 10-K extraction
schema captures much of the flavor at near-zero marginal cost.
