# Risk-factor taxonomy — observed definitions

**Status: descriptive, not authoritative.** These glosses were reverse-engineered from the 2,834
risk factors Claude Haiku labeled across the corpus. They describe what the teacher *did*, not what
the categories *should* mean. Nothing here is in `SYSTEM_PROMPT` — the prompt still ships the bare
twelve-item tuple, and changing it would invalidate every existing label and re-pay the teacher run
([D28](decisions.md#d28--the-taxonomy-ships-to-the-teacher-undefined-and-the-audit-found-it)).

**Why it exists.** The hand audit ([D26](decisions.md#d26--teacher-labels-are-audited-by-a-blind-human-sample-not-eyeballed))
asks a human whether they would pick the same category. Without written definitions that question is
unanswerable: the reviewer and the teacher disagree while both are being reasonable, and the
resulting rate measures an undefined taxonomy rather than a wrong teacher. Auditing the remaining
risks against these glosses asks a sharper question — *does the teacher follow its own rule
consistently?*

## The twelve categories

| Category | n | Covers | Does **not** cover |
|---|---:|---|---|
| `operational` | 645 | Execution and delivery failures inside the business: integration after acquisitions, restructuring, new-product execution, outsourcing, foreign operations, business-process breakdown | Systems compromised by an attacker (`cyber`); inbound supplier failure (`supply_chain`) |
| `financial` | 419 | Capital structure and treasury: liquidity, financing cost, FX, tax, pension obligations, hedging, receivables, credit | Demand or pricing for the product (`market`); economy-wide conditions (`macroeconomic`) |
| `regulatory` | 404 | Compliance with rules and agencies: approvals, licensing, rate cases, capital requirements, trade policy, data-protection and environmental statutes | Private litigation and liability (`legal`) |
| `market` | 282 | Demand side for this company's products: customer demand and concentration, sector cyclicality, commercialization, pricing power, stock-price volatility | Loss of demand specifically to rivals (`competitive`); economy-wide demand (`macroeconomic`) |
| `competitive` | 266 | Rivals and displacement: market-share pressure, new entrants, technological obsolescence, patent-cliff generic entry | General demand decline with no rival named (`market`) |
| `legal` | 185 | Litigation, liability and proceedings: product liability, warranty claims, investigations, contract disputes, and governance/anti-takeover provisions | Compliance obligations to a regulator (`regulatory`); IP suits (`intellectual_property`) |
| `macroeconomic` | 174 | Economy-wide and geopolitical conditions the company does not control: GDP, consumer spending, monetary and fiscal policy, armed conflict, political instability | Conditions in one product market (`market`) |
| `supply_chain` | 130 | Inbound dependency: suppliers, vendors, foundries, contract manufacturers, raw materials, logistics and sourcing concentration | Outbound distribution and internal execution (`operational`) |
| `cyber` | 108 | **Adversarial** compromise of systems or data: breaches, attacks, unauthorized access, data theft, plus data-security and privacy exposure | Systems failing on their own — reliability, outage, integration, internal control (`operational`) |
| `intellectual_property` | 89 | Patents and IP: protection gaps, invalidation, enforcement cost, and infringement claims in either direction | Non-IP litigation (`legal`) |
| `talent` | 77 | People: hiring, retention, key-person dependency, succession, labor relations, employee error and misconduct | — |
| `climate` | 55 | Climate change physical and transition risk, severe weather, natural disasters, and decarbonization commitments | Environmental **statutory compliance**, which the teacher files under `regulatory` |

## Tie-breaks that hold

These pairs caused real disagreements in the audit's first 20 risks. The teacher's rule is
recoverable in each case, and the reviewer should apply it:

- **`cyber` vs `operational`** — is there an adversary? Breach, attack, unauthorized access →
  `cyber`. Reliability, outage, integration, human error → `operational`. Of 63 risks mentioning
  information systems or data integrity, 40 are `cyber` and 19 `operational`, split on this line.
- **`intellectual_property` vs `legal`** — an infringement claim goes to `intellectual_property`
  (27 of 37), not `legal`, even though it is litigation.
- **`supply_chain` vs `operational`** — a named third-party supplier, vendor or manufacturer goes to
  `supply_chain` (15 of 22).
- **`regulatory` vs `climate`** — environmental compliance and remediation go to `regulatory`
  (13 of 19), not `climate`.
- **`regulatory` vs `cyber`** — data-privacy *compliance* goes to `regulatory` (9 of 15); a privacy
  *breach* goes to `cyber`.
- **`market` vs `financial`** — interest rates affecting revenue are `market`; interest rates
  affecting the company's own funding are `financial`.

## Themes with no recoverable rule

For these the teacher is not self-consistent, so no gloss can be derived from usage and a
human/teacher disagreement carries no information. **Flag these in the audit's `note` field rather
than treating either answer as correct.**

| Theme | n | Spread | Modal share |
|---|---:|---|---:|
| Pandemic / health crisis | 66 | `operational` 25 · `market` 13 · `supply_chain` 12 · `macroeconomic` 11 | 38% |
| Tariffs / trade policy | 105 | `regulatory` 40 · `macroeconomic` 33 · `supply_chain` 15 · `operational` 8 | 38% |
| Geopolitical conflict | 148 | `macroeconomic` 64 · `market` 22 · `operational` 21 · `regulatory` 19 | 43% |

A four-way split with a 38% mode is not a boundary the reviewer can be expected to guess. These are
the cases that need an *authored* rule rather than a documented one, and they are the strongest
argument for glossing the taxonomy before any future labeling run.

## The flagging criterion

A risk is marked `UNINFORMATIVE` in the audit sheet — excluded when the agreement rate is reported —
when a no-rule theme is **the risk's primary subject**, not merely a driver it cites. The
distinction matters because most risks name a macro condition somewhere in their summary, and
flagging on any mention would exclude rows whose category was never in doubt.

| | Flag? | Why |
|---|---|---|
| "Economic Downturn Impacts Revenue and Investment Returns" | yes | the downturn *is* the risk |
| "Government Debt and Political Instability Risks" | yes | political instability *is* the risk |
| "Fixed Income Security Defaults and Impairments" | no | a credit risk; the downturn is a driver, and `financial` is unambiguous |
| "Factors affecting medical cost exceeding forecasts" | no | pandemic is one driver among inflation, utilization and climate |

## Using this during the audit

The first 20 risks (BMY, AMGN, UNH, VRTX, MS) were reviewed without these definitions. The remaining
40 can be reviewed against them, which makes the two blocks separately reportable: agreement with no
stated rule, and agreement with one. The gap between those two numbers is the measurement behind
D28.
