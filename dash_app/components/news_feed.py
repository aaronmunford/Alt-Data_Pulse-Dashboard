"""
News Feed Component for Alt-Data Pulse Dashboard

Clean Bloomberg Terminal-style news panel with clickable headlines.
"""

from dash import html
import dash_bootstrap_components as dbc
from typing import Dict, List
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
    news_items = [create_news_item(h, idx) for idx, h in enumerate(headlines)]

    # Build header with optional stale indicator
    header_children = [
        html.Span("NEWS", className="news-panel-title"),
    ]

    if any_stale and show_stale_indicator:
        header_children.append(
            html.Span("CACHED", className="news-stale-badge")
        )

    return html.Div(
        [
            # Panel header
            html.Div(header_children, className="news-panel-header"),
            # Scrollable news list
            html.Div(news_items, className="news-scroll-container"),
        ],
        className="news-panel",
    )


def create_news_item(headline: Dict, index: int = 0) -> html.Div:
    """
    Create a single news item - simple timestamp + headline layout.

    Args:
        headline: Dictionary with headline data
        index: Index of the headline in the list

    Returns:
        Dash HTML component for a single news item
    """
    # Format the timestamp - show source as fallback if no timestamp
    timestamp = headline.get("timestamp")
    time_display = format_news_timestamp(timestamp)
    
    # If timestamp is empty/missing, show the source instead
    if time_display == "--:--" or not timestamp:
        source = headline.get("source", "")
        if source and source != "Unknown":
            time_display = source[:8]  # Truncate long source names
        else:
            time_display = "•"

    # Get headline text and translation status
    headline_text = headline.get("headline", "")
    is_translated = headline.get("is_translated", False)
    story_id = headline.get("story_id", str(index))
    source = headline.get("source", "")

    # Build the item content
    item_content = [
        # Timestamp with source tooltip
        html.Span(time_display, className="news-time", title=f"Source: {source}" if source else None),
        # Headline text
        html.Span(headline_text, className="news-text"),
    ]

    # Add translated indicator if applicable
    if is_translated:
        item_content.append(
            html.Span(" [TR]", className="news-translated-tag", title="Translated")
        )

    # Wrap in clickable div with pattern-matching ID
    return html.Div(
        item_content,
        className="news-item",
        id={"type": "news-headline", "index": index, "story_id": story_id},
        n_clicks=0,
    )


def create_news_unavailable_message() -> html.Div:
    """Create fallback message when news is unavailable."""
    return html.Div(
        [
            html.Div(
                [html.Span("NEWS", className="news-panel-title")],
                className="news-panel-header",
            ),
            html.Div(
                [
                    html.Div(
                        [
                            html.Div(
                                "News temporarily unavailable",
                                className="news-unavailable-text",
                            ),
                            html.Div(
                                "Waiting for LSEG connection...",
                                className="news-unavailable-subtext",
                            ),
                        ],
                        className="news-unavailable",
                    ),
                ],
                className="news-scroll-container",
            ),
        ],
        className="news-panel",
    )


def create_news_loading() -> html.Div:
    """Create loading indicator for news panel."""
    return html.Div(
        [
            html.Div(
                [html.Span("NEWS", className="news-panel-title")],
                className="news-panel-header",
            ),
            html.Div(
                [html.Div("Loading headlines...", className="news-loading")],
                className="news-scroll-container",
            ),
        ],
        className="news-panel",
    )


def create_story_modal() -> dbc.Modal:
    """
    Create the modal component for displaying full story.
    This should be added to the app layout once.
    """
    return dbc.Modal(
        [
            dbc.ModalHeader(
                dbc.ModalTitle(id="story-modal-title"),
                close_button=True,
                className="story-modal-header",
            ),
            dbc.ModalBody(
                [
                    html.Div(id="story-modal-meta", className="story-modal-meta"),
                    html.Div(id="story-modal-content", className="story-modal-content"),
                ],
            ),
        ],
        id="story-modal",
        size="lg",
        is_open=False,
        className="story-modal",
    )
