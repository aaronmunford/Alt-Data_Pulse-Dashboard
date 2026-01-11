"""
News Feed Component for Alt-Data Pulse Dashboard

Bloomberg Terminal-style news panel with scrollable headlines,
images, and translation indicators.
"""

from dash import html, dcc
import dash_bootstrap_components as dbc
from typing import Dict, List, Optional
import sys
from pathlib import Path

# Add parent directory to path for imports
parent_dir = Path(__file__).resolve().parents[1]
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

from utils import format_news_timestamp

# Language code to name mapping
LANGUAGE_NAMES = {
    "ja": "Japanese",
    "zh": "Chinese",
    "ko": "Korean",
    "de": "German",
    "fr": "French",
    "es": "Spanish",
    "pt": "Portuguese",
    "it": "Italian",
    "ru": "Russian",
    "ar": "Arabic",
}


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


def create_news_item(headline: Dict, index: int = 0) -> html.Div:
    """
    Create a single news item with Bloomberg styling.

    Args:
        headline: Dictionary with keys: headline, timestamp, story_id, source,
                  image_url, language, is_translated, story_url, is_stale
        index: Index of the headline in the list

    Returns:
        Dash HTML component for a single news item
    """
    # Format the timestamp for display
    time_display = format_news_timestamp(headline.get("timestamp"))

    # Get source badge text
    source = headline.get("source", "")

    # Get image URL if available
    image_url = headline.get("image_url")

    # Check for translation
    language = headline.get("language", "en")
    is_translated = headline.get("is_translated", False)

    # Get story URL for click handling
    story_url = headline.get("story_url")
    story_id = headline.get("story_id", "")

    # Build content elements
    content_elements = []

    # Timestamp row with language indicator
    timestamp_row = [
        html.Span(time_display, className="news-timestamp-text"),
    ]

    # Add language indicator for non-English content
    if language != "en":
        lang_name = LANGUAGE_NAMES.get(language, language.upper())
        if is_translated:
            timestamp_row.append(
                html.Span(
                    f" [Translated from {lang_name}]",
                    className="news-language-badge news-translated",
                )
            )
        else:
            timestamp_row.append(
                html.Span(
                    f" [{lang_name}]",
                    className="news-language-badge",
                )
            )

    content_elements.append(
        html.Div(timestamp_row, className="news-timestamp")
    )

    # Main content area (image + headline)
    main_content = []

    # Add thumbnail image if available
    if image_url:
        main_content.append(
            html.Div(
                html.Img(
                    src=image_url,
                    className="news-thumbnail",
                    alt="News image",
                ),
                className="news-thumbnail-container",
            )
        )

    # Headline text
    headline_text = headline.get("headline", "")
    main_content.append(
        html.Div(
            headline_text,
            className="news-headline" + (" news-headline-with-image" if image_url else ""),
        )
    )

    content_elements.append(
        html.Div(main_content, className="news-content-row")
    )

    # Source badge row
    badges = []
    if source:
        badges.append(
            html.Span(source, className="news-source-badge")
        )

    if badges:
        content_elements.append(
            html.Div(badges, className="news-badges-row")
        )

    # Wrap in clickable container if story URL available
    if story_url:
        return html.A(
            html.Div(
                content_elements,
                className="news-item-content",
            ),
            href=story_url,
            target="_blank",
            rel="noopener noreferrer",
            className="news-item news-item-clickable",
            id={"type": "news-item", "index": story_id},
        )
    else:
        # Non-clickable item (no URL available)
        return html.Div(
            content_elements,
            className="news-item",
            id={"type": "news-item", "index": story_id},
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
                [
                    html.Span("NEWS", className="news-panel-title"),
                ],
                className="news-panel-header",
            ),
            html.Div(
                [
                    html.Div(
                        [
                            html.Div(
                                "News temporarily unavailable",
                                style={
                                    "color": "#8b949e",
                                    "fontSize": "0.875rem",
                                    "marginBottom": "0.5rem",
                                },
                            ),
                            html.Div(
                                "Waiting for LSEG connection...",
                                style={
                                    "color": "#6e7681",
                                    "fontSize": "0.75rem",
                                },
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
    """
    Create loading indicator for news panel.

    Returns:
        Dash HTML component showing loading state
    """
    return html.Div(
        [
            html.Div(
                [
                    html.Span("NEWS", className="news-panel-title"),
                ],
                className="news-panel-header",
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
