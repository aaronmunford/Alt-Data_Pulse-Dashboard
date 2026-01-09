"""
Chart Building Utilities for Dash Dashboard
Bloomberg Terminal Style Plotly Charts
"""

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from datetime import datetime, timedelta

# Bloomberg Terminal Color Palette
COLORS = {
    "bg_primary": "#0a0e27",
    "bg_secondary": "#161b33",
    "grid": "#30363d",
    "text": "#e6edf3",
    "text_muted": "#8b949e",
    "green": "#3fb950",
    "red": "#f85149",
    "blue": "#58a6ff",
    "yellow": "#d29922",
    "orange": "#db6d28",
}


def create_traffic_chart(df: pd.DataFrame, date_range: str = "Last Year") -> go.Figure:
    """
    Create Foot Traffic time series chart with Bloomberg styling.

    Args:
        df: DataFrame with columns: date, total_visits, visits_7d_avg
        date_range: Filter for date range (Last 90 Days, Last 6 Months, Last Year, All Available)

    Returns:
        Plotly Figure object
    """
    if df.empty:
        return _create_empty_chart("No traffic data available")

    # Filter by date range
    df_filtered = _filter_by_date_range(df, date_range)

    if df_filtered.empty:
        return _create_empty_chart("No data in selected range")

    # Calculate traffic index (normalize to mean = 100)
    if "visits_7d_avg" in df_filtered.columns:
        y_values = df_filtered["visits_7d_avg"]
        y_label = "Store Visits (7-Day Avg)"
    elif "total_visits" in df_filtered.columns:
        y_values = df_filtered["total_visits"]
        y_label = "Store Visits (Daily)"
    else:
        return _create_empty_chart("No visit data columns found")

    # Normalize to index (mean = 100)
    mean_value = y_values.mean()
    y_index = (y_values / mean_value) * 100

    fig = go.Figure()

    # Add main line trace
    fig.add_trace(
        go.Scatter(
            x=df_filtered["date"],
            y=y_index,
            mode="lines",
            line=dict(color=COLORS["green"], width=2.5),
            name="Traffic Index",
            hovertemplate="<b>%{x|%b %d, %Y}</b><br>"
            + "Traffic Index: %{y:.1f}<br>"
            + "<extra></extra>",
        )
    )

    # Apply Bloomberg theme
    fig.update_layout(
        title={
            "text": "Foot Traffic Index (7-Day Average)",
            "font": {"family": "Space Grotesk", "size": 16, "color": COLORS["text"]},
            "x": 0,
        },
        xaxis=dict(
            title="Date",
            showgrid=True,
            gridcolor=COLORS["grid"],
            gridwidth=1,
            color=COLORS["text"],
            tickformat="%b %Y",
            tickangle=-45,
        ),
        yaxis=dict(
            title="Index (Mean = 100)",
            showgrid=True,
            gridcolor=COLORS["grid"],
            gridwidth=1,
            color=COLORS["text"],
            zeroline=False,
        ),
        plot_bgcolor=COLORS["bg_primary"],
        paper_bgcolor=COLORS["bg_primary"],
        font=dict(family="IBM Plex Mono", size=12, color=COLORS["text"]),
        hovermode="x unified",
        margin=dict(l=60, r=40, t=60, b=60),
        height=350,
    )

    return fig


def create_ticket_chart(df: pd.DataFrame, date_range: str = "Last Year") -> go.Figure:
    """
    Create Average Ticket Size time series chart with Bloomberg styling.

    Args:
        df: DataFrame with columns: date, avg_ticket_size
        date_range: Filter for date range

    Returns:
        Plotly Figure object
    """
    if df.empty:
        return _create_empty_chart("No ticket data available")

    # Filter by date range
    df_filtered = _filter_by_date_range(df, date_range)

    if df_filtered.empty:
        return _create_empty_chart("No data in selected range")

    # Get ticket size data with rolling average
    if "avg_ticket_size" not in df_filtered.columns:
        return _create_empty_chart("No avg_ticket_size column found")

    df_filtered = df_filtered.copy()
    df_filtered["ticket_7d_avg"] = (
        df_filtered["avg_ticket_size"].rolling(7, min_periods=1).mean()
    )

    fig = go.Figure()

    # Add main line trace
    fig.add_trace(
        go.Scatter(
            x=df_filtered["date"],
            y=df_filtered["ticket_7d_avg"],
            mode="lines",
            line=dict(color=COLORS["blue"], width=2.5),
            name="Avg Ticket Size",
            hovertemplate="<b>%{x|%b %d, %Y}</b><br>"
            + "Avg Ticket: $%{y:.2f}<br>"
            + "<extra></extra>",
        )
    )

    # Apply Bloomberg theme
    fig.update_layout(
        title={
            "text": "Average Ticket Size (Consumer Edge)",
            "font": {"family": "Space Grotesk", "size": 16, "color": COLORS["text"]},
            "x": 0,
        },
        xaxis=dict(
            title="Date",
            showgrid=True,
            gridcolor=COLORS["grid"],
            gridwidth=1,
            color=COLORS["text"],
            tickformat="%b %Y",
            tickangle=-45,
        ),
        yaxis=dict(
            title="Average Ticket ($)",
            showgrid=True,
            gridcolor=COLORS["grid"],
            gridwidth=1,
            color=COLORS["text"],
            zeroline=False,
            tickprefix="$",
        ),
        plot_bgcolor=COLORS["bg_primary"],
        paper_bgcolor=COLORS["bg_primary"],
        font=dict(family="IBM Plex Mono", size=12, color=COLORS["text"]),
        hovermode="x unified",
        margin=dict(l=60, r=40, t=60, b=60),
        height=350,
    )

    return fig


def create_combined_chart(
    df: pd.DataFrame, date_range: str = "Last Year"
) -> go.Figure:
    """
    Create combined chart with Traffic and Ticket Size on dual y-axes.

    Args:
        df: DataFrame with all required columns
        date_range: Filter for date range

    Returns:
        Plotly Figure with dual y-axes
    """
    if df.empty:
        return _create_empty_chart("No data available")

    df_filtered = _filter_by_date_range(df, date_range)

    if df_filtered.empty:
        return _create_empty_chart("No data in selected range")

    # Create figure with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Traffic Index (left y-axis)
    if "visits_7d_avg" in df_filtered.columns:
        y_visits = df_filtered["visits_7d_avg"]
        mean_visits = y_visits.mean()
        y_index = (y_visits / mean_visits) * 100

        fig.add_trace(
            go.Scatter(
                x=df_filtered["date"],
                y=y_index,
                mode="lines",
                line=dict(color=COLORS["green"], width=2.5),
                name="Traffic Index",
                hovertemplate="Traffic: %{y:.1f}<extra></extra>",
            ),
            secondary_y=False,
        )

    # Ticket Size (right y-axis)
    if "avg_ticket_size" in df_filtered.columns:
        df_filtered_copy = df_filtered.copy()
        df_filtered_copy["ticket_7d_avg"] = (
            df_filtered_copy["avg_ticket_size"].rolling(7, min_periods=1).mean()
        )

        fig.add_trace(
            go.Scatter(
                x=df_filtered_copy["date"],
                y=df_filtered_copy["ticket_7d_avg"],
                mode="lines",
                line=dict(color=COLORS["blue"], width=2.5),
                name="Avg Ticket",
                hovertemplate="Ticket: $%{y:.2f}<extra></extra>",
            ),
            secondary_y=True,
        )

    # Update axes
    fig.update_xaxes(
        title_text="Date",
        showgrid=True,
        gridcolor=COLORS["grid"],
        color=COLORS["text"],
        tickformat="%b %Y",
    )

    fig.update_yaxes(
        title_text="Traffic Index",
        secondary_y=False,
        showgrid=True,
        gridcolor=COLORS["grid"],
        color=COLORS["green"],
    )

    fig.update_yaxes(
        title_text="Avg Ticket ($)",
        secondary_y=True,
        showgrid=False,
        color=COLORS["blue"],
        tickprefix="$",
    )

    # Update layout
    fig.update_layout(
        title={
            "text": "Alt-Data Signals: Traffic × Ticket Size",
            "font": {"family": "Space Grotesk", "size": 16, "color": COLORS["text"]},
            "x": 0,
        },
        plot_bgcolor=COLORS["bg_primary"],
        paper_bgcolor=COLORS["bg_primary"],
        font=dict(family="IBM Plex Mono", size=12, color=COLORS["text"]),
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor="rgba(0,0,0,0)",
        ),
        margin=dict(l=60, r=60, t=80, b=60),
        height=400,
    )

    return fig


def _filter_by_date_range(df: pd.DataFrame, date_range: str) -> pd.DataFrame:
    """Filter DataFrame by date range string."""
    if df.empty:
        return df

    df = df.copy()

    # Ensure date column is datetime
    if "date" in df.columns and not pd.api.types.is_datetime64_any_dtype(df["date"]):
        df["date"] = pd.to_datetime(df["date"])

    now = datetime.now()

    if date_range == "Last 90 Days":
        cutoff = now - timedelta(days=90)
    elif date_range == "Last 6 Months":
        cutoff = now - timedelta(days=180)
    elif date_range == "Last Year":
        cutoff = now - timedelta(days=365)
    else:  # "All Available"
        return df

    return df[df["date"] >= cutoff]


def _create_empty_chart(message: str) -> go.Figure:
    """Create empty chart with message."""
    fig = go.Figure()

    fig.add_annotation(
        text=message,
        xref="paper",
        yref="paper",
        x=0.5,
        y=0.5,
        showarrow=False,
        font=dict(size=16, color=COLORS["text_muted"]),
    )

    fig.update_layout(
        plot_bgcolor=COLORS["bg_primary"],
        paper_bgcolor=COLORS["bg_primary"],
        xaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
        yaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
        height=350,
    )

    return fig
