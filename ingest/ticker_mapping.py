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

# Android app package names for QSR brands (Google Play Store)
BRAND_TO_APP_PACKAGE: Dict[str, list] = {
    "STARBUCKS": ["com.starbucks.mobilecard"],
    "MCDONALD'S": ["com.mcdonalds.app", "com.mcdonalds.mcdordering"],
    "CHIPOTLE": ["com.chipotle.ordering"],
    "DOMINO'S": ["com.dominospizza", "com.dominospizza.usa"],
    "DUNKIN": ["com.dunkinbrands.otgo"],
    "TACO BELL": ["com.yum.tacobell"],
    "WENDY'S": ["com.wendys.nutritiontool"],
    "BURGER KING": ["com.emn8.mobilem8.nativeapp.bk"],
    "SUBWAY": ["com.subway.mobile.subwayapp03"],
}

# Reverse mapping: app package -> brand
APP_PACKAGE_TO_BRAND: Dict[str, str] = {
    pkg: brand
    for brand, packages in BRAND_TO_APP_PACKAGE.items()
    for pkg in packages
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


# ============================================================================
# Revelio Labs (WRDS) Company Mappings for Hiring Data
# ============================================================================

# Revelio Labs company names that map to our canonical brands.
# Multiple variations are included to handle inconsistent naming in Revelio.
BRAND_TO_REVELIO_COMPANY: Dict[str, list] = {
    "STARBUCKS": ["Starbucks", "Starbucks Corporation", "Starbucks Coffee Company"],
    "MCDONALD'S": ["McDonald's", "McDonald's Corporation", "McDonalds"],
    "CHIPOTLE": ["Chipotle", "Chipotle Mexican Grill", "Chipotle Mexican Grill Inc"],
    "DOMINO'S": ["Domino's Pizza", "Domino's Pizza Inc", "Dominos Pizza"],
    "DUNKIN": ["Dunkin' Brands", "Dunkin' Brands Group", "Inspire Brands"],
    "TACO BELL": ["Taco Bell", "Yum! Brands"],
    "WENDY'S": ["Wendy's", "The Wendy's Company", "Wendy's International"],
    "BURGER KING": ["Burger King", "Restaurant Brands International"],
    "SUBWAY": ["Subway", "Subway IP LLC", "Doctor's Associates"],
}

# Reverse mapping: Revelio company name (lowercase) -> canonical brand
REVELIO_COMPANY_TO_BRAND: Dict[str, str] = {
    company.lower(): brand
    for brand, companies in BRAND_TO_REVELIO_COMPANY.items()
    for company in companies
}


def brand_to_revelio_companies(brand: str) -> list:
    """Return list of Revelio company names for a brand."""
    if not brand:
        return []
    return BRAND_TO_REVELIO_COMPANY.get(brand.strip().upper(), [])


def revelio_company_to_brand(company: str) -> Optional[str]:
    """Return canonical brand name for a Revelio company name."""
    if not company:
        return None
    return REVELIO_COMPANY_TO_BRAND.get(company.strip().lower())
