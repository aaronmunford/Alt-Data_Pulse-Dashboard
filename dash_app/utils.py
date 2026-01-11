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
import re
from typing import Dict, List, Optional
from datetime import datetime, timezone
from html.parser import HTMLParser

# HTML Stripper for cleaning story content
class _HTMLStripper(HTMLParser):
    """Simple HTML tag stripper."""
    def __init__(self):
        super().__init__()
        self.reset()
        self.fed = []

    def handle_data(self, data):
        self.fed.append(data)

    def get_data(self):
        return ''.join(self.fed)


def strip_html_tags(html_content: str) -> str:
    """
    Strip HTML tags from content and return clean text.

    Args:
        html_content: HTML string to clean

    Returns:
        Plain text with HTML tags removed
    """
    if not html_content:
        return ""

    # First, handle common HTML entities
    text = html_content
    text = re.sub(r'&nbsp;', ' ', text)
    text = re.sub(r'&amp;', '&', text)
    text = re.sub(r'&lt;', '<', text)
    text = re.sub(r'&gt;', '>', text)
    text = re.sub(r'&quot;', '"', text)
    text = re.sub(r'&#39;', "'", text)

    # Remove script and style elements entirely
    text = re.sub(r'<script[^>]*>.*?</script>', '', text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'<style[^>]*>.*?</style>', '', text, flags=re.DOTALL | re.IGNORECASE)

    # Replace <br> and <p> with newlines
    text = re.sub(r'<br\s*/?\s*>', '\n', text, flags=re.IGNORECASE)
    text = re.sub(r'</p>', '\n\n', text, flags=re.IGNORECASE)
    text = re.sub(r'<p[^>]*>', '', text, flags=re.IGNORECASE)

    # Replace heading tags with newlines
    text = re.sub(r'</h[1-6]>', '\n\n', text, flags=re.IGNORECASE)
    text = re.sub(r'<h[1-6][^>]*>', '', text, flags=re.IGNORECASE)

    # Remove all remaining HTML tags
    try:
        stripper = _HTMLStripper()
        stripper.feed(text)
        text = stripper.get_data()
    except Exception:
        # Fallback: simple regex strip
        text = re.sub(r'<[^>]+>', '', text)

    # Clean up whitespace
    text = re.sub(r'\n\s*\n', '\n\n', text)  # Multiple newlines to double
    text = re.sub(r'[ \t]+', ' ', text)  # Multiple spaces to single
    text = text.strip()

    return text


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
        Story content text (HTML stripped), or None if unavailable
    """
    if not story_id:
        return None

    try:
        client = get_news_client()
        story = client.get_story(story_id)
        if story and story.get("content"):
            # Strip HTML tags for clean display
            raw_content = story["content"]
            clean_content = strip_html_tags(raw_content)
            return clean_content if clean_content else None
        return None
    except Exception as e:
        print(f"Error loading story {story_id}: {e}")
        return None


def format_news_timestamp(timestamp: Optional[str]) -> str:
    """
    Format a news timestamp for display.
    Shows relative time for recent news, absolute time for older.

    Args:
        timestamp: ISO format timestamp string or pandas Timestamp

    Returns:
        Formatted string like "14:30", "2h ago", or "Jan 9"
    """
    if not timestamp:
        return "—"

    try:
        # Handle different timestamp formats
        if hasattr(timestamp, 'isoformat'):
            # It's already a datetime object (pandas Timestamp)
            dt = timestamp
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
        elif isinstance(timestamp, str):
            # Parse the ISO timestamp string
            ts = timestamp.replace("Z", "+00:00")
            # Handle pandas timestamp format with space instead of T
            ts = ts.replace(" ", "T") if "T" not in ts and " " in ts else ts
            dt = datetime.fromisoformat(ts)
        else:
            return "—"

        now = datetime.now(timezone.utc)
        diff = now - dt

        # Less than 1 minute ago
        if diff.total_seconds() < 60:
            return "now"

        # Less than 1 hour ago
        if diff.total_seconds() < 3600:
            minutes = int(diff.total_seconds() / 60)
            return f"{minutes}m"

        # Less than 24 hours ago - show time
        if diff.total_seconds() < 86400:
            hours = int(diff.total_seconds() / 3600)
            return f"{hours}h"

        # Less than 7 days ago
        if diff.days < 7:
            return f"{diff.days}d"

        # Older than a week - show date
        return dt.strftime("%b %d")

    except Exception as e:
        # If all parsing fails, return a dash
        return "—"


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
