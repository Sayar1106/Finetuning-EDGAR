"""Build the bounded filing excerpt that the student model sees as input.

Two constraints drive this module:

1. **Sequence length.** Risk-factor sections alone run 36K-146K characters; a full 10-K is far past
   what an 8B model can be fine-tuned on at a sane sequence length. The input must be an excerpt.

2. **Label grounding.** A label the input doesn't support is unlearnable, and worse, it teaches the
   model to hallucinate confidently. Two consequences:
   - The teacher labels *this excerpt*, not the full filing (see src/labels/teacher.py), so every
     risk factor in the target is present in the input.
   - MD&A is windowed around the XBRL figures rather than truncated from the top, and
     `check_grounding()` reports which numeric labels actually appear in the text. Ungrounded
     numerics are a dataset-quality signal, not something to paper over.
"""

from __future__ import annotations

from src.data.schema import FilingSections, FinancialFacts

BUSINESS_CHARS = 3_000
MDNA_CHARS = 6_000
RISK_CHARS = 12_000
STATEMENT_CHARS = 3_500  # per statement; these are dense and carry most numeric ground truth

# Filings report dollar figures scaled -- "in millions" or "in thousands" -- so an exact XBRL value
# like 416161000000 never appears verbatim. These are the renderings actually seen in practice.
_SCALES = (1, 1_000, 1_000_000)


def numeric_variants(value: float | None) -> list[str]:
    """String renderings of `value` a filing might plausibly use, for grounding checks."""
    if value is None:
        return []

    variants: set[str] = set()
    for scale in _SCALES:
        scaled = value / scale
        if abs(scaled) < 0.01:
            continue
        if abs(scaled - round(scaled)) < 1e-6:
            variants.add(f"{round(scaled):,}")
        # Filings also write "416.2 billion" / "7.46" style figures.
        variants.add(f"{scaled:,.1f}")
        variants.add(f"{scaled:,.2f}")

    # Trailing ".0" is usually dropped in prose.
    variants = {v[:-2] if v.endswith(".0") else v for v in variants}

    # Drop variants too short to be evidence. Scaling 5,000 down to thousands yields "5", which
    # matches the "5" inside "2025" and reports the label as grounded when it is not. Requiring
    # three digits keeps real figures and removes the false positives.
    return sorted(v for v in variants if sum(c.isdigit() for c in v) >= 3)


def _window_around_figures(text: str, facts: FinancialFacts, budget: int) -> str:
    """Return a `budget`-sized window of `text` centered on the first XBRL figure found in it.

    Falls back to the head of the section when no figure is present -- MD&A opens with an overview
    that is still useful context, so an ungrounded window beats an empty one.
    """
    if len(text) <= budget:
        return text

    anchors = []
    for value in (facts.revenue, facts.net_income, facts.total_assets):
        for variant in numeric_variants(value):
            idx = text.find(variant)
            if idx != -1:
                anchors.append(idx)

    if not anchors:
        return text[:budget]

    start = max(0, min(anchors) - budget // 4)
    return text[start : start + budget]


def build_excerpt(sections: FilingSections) -> str:
    """Assemble the bounded, plain-text filing excerpt used as model input."""
    business = (sections.business or "").strip()
    mdna = (sections.management_discussion or "").strip()
    risk = (sections.risk_factors or "").strip()

    parts = [
        f"Company: {sections.company_name}",
        f"Ticker: {sections.ticker}",
        f"Form: {sections.form_type}",
        f"Fiscal year: {sections.fiscal_year}",
    ]
    if business:
        parts.append("\n## Item 1. Business\n" + business[:BUSINESS_CHARS])
    # Statements come before the narrative sections: they are where the numeric answers live, and
    # putting them first keeps them intact if anything downstream truncates.
    if sections.income_statement:
        parts.append("\n" + sections.income_statement[:STATEMENT_CHARS])
    if sections.balance_sheet:
        parts.append("\n" + sections.balance_sheet[:STATEMENT_CHARS])
    if mdna:
        parts.append(
            "\n## Item 7. Management's Discussion and Analysis\n"
            + _window_around_figures(mdna, sections.financials, MDNA_CHARS)
        )
    if risk:
        parts.append("\n## Item 1A. Risk Factors\n" + risk[:RISK_CHARS])

    return "\n".join(parts)


def check_grounding(excerpt: str, facts: FinancialFacts) -> dict[str, bool]:
    """For each non-null numeric fact, whether some rendering of it appears in the excerpt.

    A `False` here means the label cannot be derived from the input. Reported in aggregate by
    src/labels/build.py rather than silently dropped, so the dataset's real ceiling is visible.
    """
    grounded: dict[str, bool] = {}
    for field in ("revenue", "net_income", "total_assets", "eps_diluted"):
        value = getattr(facts, field)
        if value is None:
            continue
        grounded[field] = any(v in excerpt for v in numeric_variants(value))
    return grounded
