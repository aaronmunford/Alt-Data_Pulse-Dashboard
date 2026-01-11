"""
Mapping helpers for aligning brand names with LSEG/Refinitiv identifiers.
This keeps ticker logic centralized so the rest of the pipeline can stay simple.
"""

from typing import Dict, Optional

# Canonical brand names used by the predictor/dashboard.
# Use uppercase keys to make lookup tolerant of casing differences.
BRAND_TO_TICKER: Dict[str, Optional[str]] = {
    "STARBUCKS": "SBUX.O",
    "MCDONALD'S": "MCD",
    "CHIPOTLE": "CMG",
    "DOMINO'S": "DPZ",
    "DUNKIN": None,  # Private/Delisted - no live LSEG consensus
}

TICKER_TO_BRAND: Dict[str, str] = {
    ticker: brand for brand, ticker in BRAND_TO_TICKER.items() if ticker
}

# PermIDs can be useful for future LSEG queries or entity matching.
TICKER_TO_PERMID: Dict[str, str] = {
    "SBUX.O": "4295905573",
    "MCD": "4295904557",
    "CMG": "4297089838",
    "DPZ": "5037915697",
}

# Simple fiscal calendar tags for quick reference in UI/logging.
FISCAL_CALENDAR_LABEL: Dict[str, str] = {
    "SBUX.O": "Oct FYE",
    "MCD": "Dec FYE",
    "CMG": "Dec FYE",
    "DPZ": "Dec FYE",
}


def brand_to_ticker(brand: str) -> Optional[str]:
    """Return the Refinitiv ticker for a brand, or None if unsupported."""
    if not brand:
        return None
    return BRAND_TO_TICKER.get(brand.strip().upper())


def ticker_to_brand(ticker: str) -> Optional[str]:
    """Return the canonical brand name for a ticker."""
    if not ticker:
        return None
    return TICKER_TO_BRAND.get(ticker.strip().upper())


def get_fiscal_period_params(ticker: str, quarter: Optional[str]) -> Dict:
    """
    Build LSEG query parameters and metadata for a given fiscal quarter.

    Args:
        ticker: Refinitiv ticker (e.g., SBUX.O).
        quarter: Fiscal quarter label like "2025_Q1".

    Returns:
        Dictionary with:
        - parameters: LSEG query params for fiscal-quarter time series
        - period_label: original quarter label for filtering
        - fiscal_calendar: human-readable fiscal year-end label
    """
    period_label = quarter
    fiscal_calendar = FISCAL_CALENDAR_LABEL.get(ticker, "Dec FYE")

    # LSEG parameters request a rolling window of fiscal quarters.
    # Filtering to the exact quarter happens in the client (if available in data).
    parameters = {
        "SDate": "0",
        "EDate": "-8",
        "Frq": "FQ",
        "Curn": "USD",
    }

    return {
        "parameters": parameters,
        "period_label": period_label,
        "fiscal_calendar": fiscal_calendar,
    }


def supported_tickers() -> list[str]:
    """Return tickers that have live consensus support."""
    return sorted({ticker for ticker in BRAND_TO_TICKER.values() if ticker})
