"""
Data Loading Utilities for Dash App
"""
import sys
from pathlib import Path

# Add parent directory to path for imports
try:
    parent_dir = Path(__file__).resolve().parents[1]
    if parent_dir.exists():
        sys.path.insert(0, str(parent_dir))
    else:
        print(f"Warning: parent directory not found for imports: {parent_dir}", file=sys.stderr)
except Exception as exc:
    print(f"Warning: unable to set import path for dash_app: {exc}", file=sys.stderr)

from ingest.predictor import RevenuePredictor
from ingest.news_client import LSEGNewsClient
import pandas as pd
from typing import Dict, List, Optional
from datetime import datetime, timezone

# Singleton predictor instance (process-local; run workers=1 to avoid shared-state issues)
_predictor = None
_news_client = None


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

def get_news_client() -> LSEGNewsClient:
    """Get or create LSEGNewsClient singleton."""
    global _news_client
    if _news_client is None:
        _news_client = LSEGNewsClient()
    return _news_client


def load_brand_news(brand: str, count: int = 15) -> List[Dict]:
    """
    Load news headlines for a brand.

    Args:
        brand: Brand name (e.g., "STARBUCKS")
        count: Number of headlines to fetch

    Returns:
        List of dicts with keys: headline, timestamp, story_id, source, is_stale
    """
    try:
        client = get_news_client()
        headlines = client.get_headlines_for_brand(brand, count=count)
        return headlines
    except Exception as e:
        print(f"Error loading news for {brand}: {e}")
        return []


def load_story_content(story_id: str) -> Optional[str]:
    """
    Load full story content by story ID.

    Args:
        story_id: The LSEG story identifier

    Returns:
        Story content text, or None if unavailable
    """
    if not story_id:
        return None

    try:
        client = get_news_client()
        story = client.get_story(story_id)
        if story and story.get("content"):
            return story["content"]
        return None
    except Exception as e:
        print(f"Error loading story {story_id}: {e}")
        return None


def format_news_timestamp(timestamp: Optional[str]) -> str:
    """
    Format a news timestamp for display.
    Shows relative time for recent news, absolute time for older.

    Args:
        timestamp: ISO format timestamp string

    Returns:
        Formatted string like "2h ago", "Yesterday 14:30", or "Jan 9 14:30"
    """
    if not timestamp:
        return ""

    try:
        # Parse the ISO timestamp
        dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
        now = datetime.now(timezone.utc)
        diff = now - dt

        # Less than 1 minute ago
        if diff.total_seconds() < 60:
            return "Just now"

        # Less than 1 hour ago
        if diff.total_seconds() < 3600:
            minutes = int(diff.total_seconds() / 60)
            return f"{minutes}m ago"

        # Less than 24 hours ago
        if diff.total_seconds() < 86400:
            hours = int(diff.total_seconds() / 3600)
            return f"{hours}h ago"

        # Less than 7 days ago
        if diff.days < 7:
            if diff.days == 1:
                return f"Yesterday {dt.strftime('%H:%M')}"
            return f"{diff.days}d ago"

        # Older than a week - show date
        return dt.strftime("%b %d %H:%M")

    except Exception:
        return ""


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
