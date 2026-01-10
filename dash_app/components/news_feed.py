"""
News Feed Component for Alt-Data Pulse Dashboard

Bloomberg Terminal-style news panel with scrollable headlines.
"""

from dash import html
from typing import Dict, List, Optional
import sys
from pathlib import Path

# Add parent directory to path for imports
parent_dir = Path(__file__).resolve().parents[1]
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

from utils import format_news_timestamp


def create_news_feed_panel(
    headlines: List[Dict],
    is_loading: bool = False,
    show_stale_indicator: bool = False,
) -> html.Div:
    """
    Create Bloomberg-style news feed panel.

    Args:
        headlines: List of headline dictionaries
        is_loading: Whether news is currently loading
        show_stale_indicator: Whether to show a stale cache indicator

    Returns:
        Dash HTML component for the news panel
    """
    if is_loading:
        return create_news_loading()

    if not headlines:
        return create_news_unavailable_message()

    # Check if any headlines are stale
    any_stale = any(h.get("is_stale", False) for h in headlines)

    # Build the news items
    news_items = [create_news_item(h) for h in headlines]

    # Build header with optional stale indicator
    header_children = [
        html.Span("NEWS", className="news-panel-title"),
    ]

    if any_stale and show_stale_indicator:
        header_children.append(
            html.Span(
                "CACHED",
                className="news-stale-badge",
            )
        )

    return html.Div(
        [
            # Panel header
            html.Div(
                header_children,
                className="news-panel-header",
            ),
            # Scrollable news list
            html.Div(
                news_items,
                className="news-scroll-container",
            ),
        ],
        className="news-panel",
    )


def create_news_item(headline: Dict) -> html.Div:
    """
    Create a single news item with Bloomberg styling.

    Args:
        headline: Dictionary with keys: headline, timestamp, story_id, source, is_stale

    Returns:
        Dash HTML component for a single news item
    """
    # Format the timestamp for display
    time_display = format_news_timestamp(headline.get("timestamp"))

    # Get source badge text
    source = headline.get("source", "")

    # Build the news item
    return html.Div(
        [
            # Timestamp row
            html.Div(
                time_display,
                className="news-timestamp",
            ),
            # Headline text
            html.Div(
                headline.get("headline", ""),
                className="news-headline",
            ),
            # Source badge
            html.Span(
                source,
                className="news-source-badge",
            ) if source else None,
        ],
        className="news-item",
        id={"type": "news-item", "index": headline.get("story_id", "")},
    )


def create_news_unavailable_message() -> html.Div:
    """
    Create fallback message when news is unavailable.

    Returns:
        Dash HTML component showing unavailable message
    """
    return html.Div(
        [
            html.Div(
                "NEWS",
                className="news-panel-title",
                style={"marginBottom": "1rem"},
            ),
            html.Div(
                [
                    html.Div(
                        "News temporarily unavailable",
                        style={
                            "color": "#6e7681",
                            "fontSize": "0.875rem",
                            "marginBottom": "0.5rem",
                        },
                    ),
                    html.Div(
                        "Check LSEG Workspace connection",
                        style={
                            "color": "#484f58",
                            "fontSize": "0.75rem",
                        },
                    ),
                ],
                className="news-unavailable",
            ),
        ],
        className="news-panel",
    )


def create_news_loading() -> html.Div:
    """
    Create loading indicator for news panel.

    Returns:
        Dash HTML component showing loading state
    """
    return html.Div(
        [
            html.Div(
                "NEWS",
                className="news-panel-title",
            ),
            html.Div(
                [
                    html.Div(
                        "Loading headlines...",
                        className="news-loading",
                    ),
                ],
                className="news-scroll-container",
            ),
        ],
        className="news-panel",
    )
