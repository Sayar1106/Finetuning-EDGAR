"""The company universe this project pulls filings for.

A fixed, version-controlled list (not a live index pull) so the dataset is reproducible run to
run. Mix of large-cap and mid-cap tickers across sectors for diversity in filing style and length.
Company-level (not filing-level) train/val/test splits are derived from this list in
src/labels/splits.py.
"""

from __future__ import annotations

# ~200 tickers spanning sector and market-cap diversity. Extend this list as needed; downstream
# code should never assume a fixed length.
TICKERS: list[str] = [
    # Technology
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "AVGO", "CRM", "ORCL", "ADBE",
    "CSCO", "INTC", "AMD", "QCOM", "TXN", "IBM", "NOW", "INTU", "PANW", "SNPS",
    "CDNS", "ADI", "MU", "LRCX", "KLAC", "APH", "ANSS", "FTNT", "PAYX", "ADSK",
    # Financials
    "JPM", "BAC", "WFC", "GS", "MS", "C", "SCHW", "BLK", "SPGI", "ICE",
    "CME", "AXP", "USB", "PNC", "TFC", "AON", "MMC", "AIG", "MET", "PRU",
    # Healthcare
    "UNH", "JNJ", "LLY", "PFE", "ABBV", "MRK", "TMO", "ABT", "DHR", "BMY",
    "AMGN", "GILD", "CVS", "CI", "ELV", "HUM", "ISRG", "VRTX", "REGN", "ZTS",
    # Consumer
    "WMT", "PG", "KO", "PEP", "COST", "HD", "MCD", "NKE", "SBUX", "TGT",
    "LOW", "TJX", "BKNG", "CMG", "MAR", "YUM", "DG", "ROST", "EL", "CL",
    # Industrials
    "CAT", "HON", "UNP", "UPS", "BA", "GE", "LMT", "RTX", "DE", "MMM",
    "EMR", "ETN", "ITW", "NSC", "CSX", "FDX", "PH", "GD", "NOC", "WM",
    # Energy & Materials
    "XOM", "CVX", "COP", "SLB", "EOG", "PSX", "MPC", "OXY", "WMB", "LIN",
    "APD", "ECL", "SHW", "NEM", "FCX", "DOW", "DD", "PPG", "ALB", "VMC",
    # Communications & Media
    "DIS", "CMCSA", "NFLX", "T", "VZ", "TMUS", "CHTR", "WBD", "OMC", "EA",
    # Utilities & Real Estate
    "NEE", "DUK", "SO", "D", "AEP", "EXC", "SRE", "AMT", "PLD", "EQIX",
    "PSA", "O", "SPG", "WELL", "AVB",
    # Mid-cap diversity (smaller filers, more variable filing quality/length)
    "ETSY", "RGEN", "FIVE", "WSM", "POOL", "SAIA", "CROX", "BLDR", "AXON", "TTEK",
    "MASI", "CHE", "FN", "ONTO", "MKTX", "SSD", "EXPO", "UFPI", "LSTR", "MEDP",
    "CACC", "WING", "CVLT", "POWI", "ENSG", "CWST", "NVEE", "SITE", "RGLD", "TREX",
]

# EDGAR's ticker->CIK map (https://www.sec.gov/files/company_tickers.json) lists only a company's
# *current* ticker, so a symbol that stops being current is simply absent and `Company(ticker)`
# raises "Company not found". Two ways that happens, both seen in this universe:
#
#   - rename: the company still files, under a new symbol (MMC -> MRSH)
#   - acquisition: the company is delisted and has no ticker at all, but its historical filings
#     remain valid (ANSS, acquired by Synopsys; last 10-K is FY2024)
#
# A CIK is permanent through both, so pin the lookup to it rather than chasing symbols. Each CIK
# below was resolved by company name against SEC's map and confirmed to have 10-K filings -- not
# recalled from memory. A fuzzy ticker match is how you end up ingesting Mag Mile Capital ('MMCP',
# edgartools' suggestion for MMC) and calling it Marsh & McLennan.
#
# The keys stay the original tickers deliberately. src/labels/splits.py assigns companies to
# train/val/test by hashing the ticker, so renaming a key would move that company to a different
# split and invalidate comparisons against every earlier run.
CIK_OVERRIDES: dict[str, int] = {
    "MMC": 62709,  # MARSH & MCLENNAN COMPANIES, INC. -- now trades as MRSH
    "ANSS": 1013462,  # ANSYS INC -- delisted after the Synopsys acquisition; final 10-K is FY2024
}
