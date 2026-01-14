"""
Data Loading Utilities for Dash App
"""
import sys
from pathlib import Path
import re
from bs4 import BeautifulSoup

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


def load_brand_hiring_data(brand: str) -> pd.DataFrame:
    """
    Load monthly hiring trend data for a brand.

    Returns:
        DataFrame with columns: date, headcount, inflows, outflows,
        hiring_velocity, attrition_rate, net_hiring
    """
    try:
        predictor = get_predictor()
        df = predictor.get_hiring_trend_data(brand)
        return df
    except Exception as e:
        print(f"Error loading hiring data for {brand}: {e}")
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

def _clean_html_content(raw_html: str) -> str:
    """
    Clean HTML content to plain text, preserving paragraphs and breaks.
    Handles complex HTML like social media embeds, tables, base64 images.
    """
    if not raw_html:
        return ""
    
    try:
        soup = BeautifulSoup(raw_html, "html.parser")
        
        # Remove script, style, and meta elements entirely
        for element in soup(["script", "style", "meta", "link", "noscript"]):
            element.decompose()
        
        # Remove image elements (especially base64 encoded ones that add noise)
        for img in soup.find_all("img"):
            img.decompose()
        
        # Remove social media specific elements that don't contain useful text
        for element in soup.find_all(class_=re.compile(r'social-avatar|social-verified')):
            element.decompose()
        
        # Replace <br> with newline
        for br in soup.find_all("br"):
            br.replace_with("\n")

        # Replace block elements with newlines for readability
        for tag in soup.find_all(["p", "div", "h1", "h2", "h3", "h4", "h5", "h6", "li", "tr"]):
            tag.insert_before("\n")
            tag.insert_after("\n")
        
        # Get text content
        text = soup.get_text(separator=" ")
        
        # Clean up whitespace
        # Replace tabs and multiple spaces with single space
        text = re.sub(r'[\t ]+', ' ', text)
        
        # Split into lines, strip each, remove empty lines
        lines = [line.strip() for line in text.splitlines()]
        lines = [line for line in lines if line]
        
        # Join with single newlines
        text = '\n'.join(lines)
        
        # Limit consecutive newlines to max 2
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # Remove any leftover HTML entities
        text = re.sub(r'&[a-z]+;', ' ', text, flags=re.IGNORECASE)
        text = re.sub(r'&#?\w+;', '', text)
        
        # Final cleanup
        text = text.strip()
        
        # If result is too short or looks like garbage, return a friendly message
        if len(text) < 20 or text.count(' ') < 3:
            return ""
        
        return text
        
    except Exception as e:
        print(f"Error cleaning HTML: {e}")
        # Try a simple fallback: strip all tags
        try:
            clean = re.sub(r'<[^>]+>', ' ', raw_html)
            clean = re.sub(r'\s+', ' ', clean).strip()
            return clean if len(clean) > 20 else ""
        except:
            return ""


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
            return _clean_html_content(story["content"])
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
        return "--:--"

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
                return f"Yest {dt.strftime('%H:%M')}"
            return f"{diff.days}d ago"

        # Older than a week - show date
        return dt.strftime("%b %d")

    except Exception:
        # Fallback: try to extract time part from string if it looks like ISO
        try:
            return timestamp.split("T")[1][:5]
        except:
            return timestamp[:10]  # Return date part only



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


def calculate_hiring_kpis(df: pd.DataFrame) -> Dict:
    """
    Calculate hiring KPIs from hiring data.

    Args:
        df: DataFrame with hiring data for a single brand

    Returns:
        Dict with keys: total_job_openings, job_openings_change_6m, job_openings_change_1y,
        total_employees, employee_change_6m, employee_change_1y, employee_change_2y,
        net_hiring_trend, hiring_velocity_avg
    """
    if df.empty:
        return {
            "total_job_openings": 0,
            "job_openings_change_6m": 0.0,
            "job_openings_change_1y": 0.0,
            "total_employees": 0,
            "employee_change_6m": 0.0,
            "employee_change_1y": 0.0,
            "employee_change_2y": 0.0,
            "net_hiring_trend": "neutral",
            "hiring_velocity_avg": 0.0,
            "latest_inflows": 0,
            "latest_outflows": 0,
        }

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")

    # Get latest values
    latest = df.iloc[-1]

    # Use unique_active_job_count if available (LinkUp), else use headcount
    job_col = "unique_active_job_count" if "unique_active_job_count" in df.columns else "headcount"
    total_job_openings = float(latest.get(job_col, 0) or 0)
    total_employees = float(latest.get("headcount", 0) or 0)

    # Calculate period changes
    latest_date = df["date"].max()

    def get_pct_change(months_ago: int, column: str) -> float:
        """Calculate percentage change from N months ago."""
        target_date = latest_date - pd.DateOffset(months=months_ago)
        historical = df[df["date"] <= target_date]
        if historical.empty:
            return 0.0
        historical_value = float(historical.iloc[-1].get(column, 0) or 0)
        current_value = float(latest.get(column, 0) or 0)
        if historical_value == 0:
            return 0.0
        return ((current_value - historical_value) / historical_value) * 100

    job_openings_change_6m = get_pct_change(6, job_col)
    job_openings_change_1y = get_pct_change(12, job_col)

    employee_change_6m = get_pct_change(6, "headcount")
    employee_change_1y = get_pct_change(12, "headcount")
    employee_change_2y = get_pct_change(24, "headcount")

    # Calculate averages
    hiring_velocity_avg = float(df["hiring_velocity"].mean()) if "hiring_velocity" in df.columns else 0.0

    # Determine net hiring trend (last 3 months)
    recent = df.tail(3)
    if "net_hiring" in recent.columns:
        recent_net = recent["net_hiring"].sum()
        if recent_net > 100:
            net_hiring_trend = "increasing"
        elif recent_net < -100:
            net_hiring_trend = "decreasing"
        else:
            net_hiring_trend = "stable"
    else:
        net_hiring_trend = "neutral"

    return {
        "total_job_openings": total_job_openings,
        "job_openings_change_6m": job_openings_change_6m,
        "job_openings_change_1y": job_openings_change_1y,
        "total_employees": total_employees,
        "employee_change_6m": employee_change_6m,
        "employee_change_1y": employee_change_1y,
        "employee_change_2y": employee_change_2y,
        "net_hiring_trend": net_hiring_trend,
        "hiring_velocity_avg": hiring_velocity_avg,
        "latest_inflows": float(latest.get("inflows", 0) or 0),
        "latest_outflows": float(latest.get("outflows", 0) or 0),
    }


def format_hiring_number(value: float) -> str:
    """Format large numbers with K/M suffix."""
    if value >= 1_000_000:
        return f"{value/1_000_000:.1f}M"
    elif value >= 1_000:
        return f"{value/1_000:.1f}K"
    else:
        return f"{value:,.0f}"


def format_pct_change(value: float) -> str:
    """Format percentage change with sign and color indicator."""
    sign = "+" if value >= 0 else ""
    return f"{sign}{value:.1f}%"
