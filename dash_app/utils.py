"""
Data Loading Utilities for Dash App
"""
import sys
from pathlib import Path

# Add parent directory to path for imports
parent_dir = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(parent_dir))

from ingest.predictor import RevenuePredictor
import pandas as pd
from typing import Dict, Optional
from datetime import datetime

# Singleton predictor instance
_predictor = None

def get_predictor() -> RevenuePredictor:
    """Get or create RevenuePredictor singleton."""
    global _predictor
    if _predictor is None:
        _predictor = RevenuePredictor()
    return _predictor

def load_brand_signal(brand: str) -> Dict:
    """
    Load revenue signal for a brand.

    Returns:
        Dict with keys: brand, quarter, predicted_revenue, wall_street_consensus,
        delta_pct, delta_direction, signal_strength, correlation, etc.
    """
    try:
        predictor = get_predictor()
        signal = predictor.predict_revenue(brand)

        if 'error' in signal:
            return {
                'error': signal['error'],
                'brand': brand,
            }

        return signal
    except Exception as e:
        return {
            'error': str(e),
            'brand': brand,
        }

def load_brand_trend_data(brand: str) -> pd.DataFrame:
    """
    Load daily trend data for a brand.

    Returns:
        DataFrame with columns: date, spend, transactions, avg_ticket_size,
        spend_7d_avg, transactions_7d_avg, total_visits (if available),
        visits_7d_avg (if available), etc.
    """
    try:
        predictor = get_predictor()
        df = predictor.get_trend_data(brand)
        return df
    except Exception as e:
        print(f"Error loading trend data for {brand}: {e}")
        return pd.DataFrame()

def get_available_brands():
    """Get list of available brands."""
    return ["STARBUCKS", "MCDONALD'S", "CHIPOTLE"]

def format_currency(value: float) -> str:
    """Format value as currency in billions."""
    if value >= 1e9:
        return f"${value/1e9:.2f}B"
    elif value >= 1e6:
        return f"${value/1e6:.1f}M"
    else:
        return f"${value:,.0f}"

def format_delta(delta_pct: float) -> str:
    """Format delta percentage with + or - sign."""
    sign = "+" if delta_pct >= 0 else ""
    return f"{sign}{delta_pct:.2f}%"

def get_delta_color(delta_pct: float) -> str:
    """Get color for delta based on positive/negative."""
    return "#3fb950" if delta_pct >= 0 else "#f85149"

def get_signal_color(signal_strength: str) -> str:
    """Get color for signal strength."""
    colors = {
        "Very High": "#3fb950",
        "High": "#58a6ff",
        "Medium": "#d29922",
        "Low": "#f85149",
    }
    return colors.get(signal_strength, "#8b949e")

def refresh_predictor():
    """Force refresh of predictor data."""
    global _predictor
    _predictor = None
    return get_predictor()
