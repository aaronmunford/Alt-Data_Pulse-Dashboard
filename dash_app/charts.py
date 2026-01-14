"""
Chart Building Utilities for Dash Dashboard
Bloomberg Terminal Style Plotly Charts
"""

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from datetime import timedelta

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


def create_app_engagement_chart(
    df: pd.DataFrame, date_range: str = "Last Year"
) -> go.Figure:
    """
    Create App Engagement chart with DAU and Installs on dual y-axes.
    
    Args:
        df: DataFrame with columns: date, dau, installs (optional)
        date_range: Filter for date range
    
    Returns:
        Plotly Figure with app engagement data
    """
    if df.empty:
        return _create_empty_chart("No app engagement data available")

    # Check for required columns
    has_dau = 'dau' in df.columns
    has_installs = 'installs' in df.columns
    
    if not has_dau and not has_installs:
        return _create_empty_chart("No DAU or installs data available")

    df_filtered = _filter_by_date_range(df, date_range)

    if df_filtered.empty:
        return _create_empty_chart("No data in selected range")

    # Create figure with secondary y-axis if both metrics available
    if has_dau and has_installs:
        fig = make_subplots(specs=[[{"secondary_y": True}]])
    else:
        fig = go.Figure()

    df_filtered = df_filtered.copy()

    # DAU (primary - left y-axis)
    if has_dau:
        # Calculate 7-day rolling average
        df_filtered['dau_7d_avg'] = df_filtered['dau'].rolling(7, min_periods=1).mean()
        
        # Calculate DAU index for easier comparison (mean = 100)
        mean_dau = df_filtered['dau_7d_avg'].mean()
        if mean_dau > 0:
            df_filtered['dau_index'] = (df_filtered['dau_7d_avg'] / mean_dau) * 100
        else:
            df_filtered['dau_index'] = 0
        
        fig.add_trace(
            go.Scatter(
                x=df_filtered["date"],
                y=df_filtered["dau_index"],
                mode="lines",
                line=dict(color=COLORS["orange"], width=2.5),
                name="DAU Index",
                hovertemplate="<b>%{x|%b %d, %Y}</b><br>"
                + "DAU Index: %{y:.1f}<br>"
                + "<extra></extra>",
            ),
            secondary_y=False if has_installs else None,
        )

    # Installs (secondary - right y-axis)
    if has_installs:
        df_filtered['installs_7d_avg'] = df_filtered['installs'].rolling(7, min_periods=1).mean()
        
        fig.add_trace(
            go.Scatter(
                x=df_filtered["date"],
                y=df_filtered["installs_7d_avg"],
                mode="lines",
                line=dict(color=COLORS["yellow"], width=2),
                name="Installs (7d avg)",
                hovertemplate="<b>%{x|%b %d, %Y}</b><br>"
                + "Installs: %{y:,.0f}<br>"
                + "<extra></extra>",
            ),
            secondary_y=True if has_dau else None,
        )

    # Update axes
    if has_dau and has_installs:
        fig.update_xaxes(
            title_text="Date",
            showgrid=True,
            gridcolor=COLORS["grid"],
            color=COLORS["text"],
            tickformat="%b %Y",
        )
        fig.update_yaxes(
            title_text="DAU Index (Mean=100)",
            secondary_y=False,
            showgrid=True,
            gridcolor=COLORS["grid"],
            color=COLORS["orange"],
        )
        fig.update_yaxes(
            title_text="Daily Installs",
            secondary_y=True,
            showgrid=False,
            color=COLORS["yellow"],
        )
    else:
        fig.update_xaxes(
            title_text="Date",
            showgrid=True,
            gridcolor=COLORS["grid"],
            color=COLORS["text"],
            tickformat="%b %Y",
        )
        fig.update_yaxes(
            title_text="DAU Index" if has_dau else "Daily Installs",
            showgrid=True,
            gridcolor=COLORS["grid"],
            color=COLORS["orange"] if has_dau else COLORS["yellow"],
        )

    # Update layout
    fig.update_layout(
        title={
            "text": "📱 App Engagement (Similarweb)",
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
        height=350,
    )

    return fig


def create_hiring_chart(
    df: pd.DataFrame, date_range: str = "Last Year"
) -> go.Figure:
    """
    Create Hiring Trends chart with Headcount and Hiring Velocity on dual y-axes.

    Args:
        df: DataFrame with columns: date, headcount, hiring_velocity, inflows, net_hiring
        date_range: Filter for date range

    Returns:
        Plotly Figure with hiring data
    """
    if df.empty:
        return _create_empty_chart("No hiring data available")

    # Check for required columns
    has_headcount = 'headcount' in df.columns
    has_velocity = 'hiring_velocity' in df.columns
    has_inflows = 'inflows' in df.columns

    if not has_headcount and not has_velocity and not has_inflows:
        return _create_empty_chart("No hiring metrics available")

    df_filtered = _filter_by_date_range(df, date_range)

    if df_filtered.empty:
        return _create_empty_chart("No data in selected range")

    # Create figure with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    df_filtered = df_filtered.copy()

    # Headcount (primary - left y-axis, bar chart)
    if has_headcount and df_filtered['headcount'].sum() > 0:
        fig.add_trace(
            go.Bar(
                x=df_filtered["date"],
                y=df_filtered["headcount"],
                name="Headcount",
                marker_color=COLORS["blue"],
                opacity=0.6,
                hovertemplate="<b>%{x|%b %Y}</b><br>"
                + "Headcount: %{y:,.0f}<br>"
                + "<extra></extra>",
            ),
            secondary_y=False,
        )

    # Hiring Velocity (secondary - right y-axis, line)
    if has_velocity:
        fig.add_trace(
            go.Scatter(
                x=df_filtered["date"],
                y=df_filtered["hiring_velocity"],
                mode="lines+markers",
                line=dict(color=COLORS["green"], width=2.5),
                marker=dict(size=6, color=COLORS["green"]),
                name="Hiring Velocity (%)",
                hovertemplate="<b>%{x|%b %Y}</b><br>"
                + "Hiring Velocity: %{y:+.1f}%<br>"
                + "<extra></extra>",
            ),
            secondary_y=True,
        )

    # Net Hiring (additional line on right y-axis)
    if 'net_hiring' in df_filtered.columns:
        fig.add_trace(
            go.Scatter(
                x=df_filtered["date"],
                y=df_filtered["net_hiring"],
                mode="lines",
                line=dict(color=COLORS["yellow"], width=2, dash="dot"),
                name="Net Hiring",
                hovertemplate="<b>%{x|%b %Y}</b><br>"
                + "Net Hiring: %{y:+,.0f}<br>"
                + "<extra></extra>",
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

    if has_headcount and df_filtered['headcount'].sum() > 0:
        fig.update_yaxes(
            title_text="Headcount",
            secondary_y=False,
            showgrid=True,
            gridcolor=COLORS["grid"],
            color=COLORS["blue"],
            tickformat=",",
        )
    else:
        fig.update_yaxes(
            title_text="",
            secondary_y=False,
            showgrid=True,
            gridcolor=COLORS["grid"],
        )

    fig.update_yaxes(
        title_text="Hiring Velocity (MoM %)",
        secondary_y=True,
        showgrid=False,
        color=COLORS["green"],
        ticksuffix="%",
    )

    # Update layout
    fig.update_layout(
        title={
            "text": "Hiring Trends (Revelio Labs)",
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
        height=350,
        barmode="overlay",
    )

    return fig


def _filter_by_date_range(df: pd.DataFrame, date_range: str) -> pd.DataFrame:
    """Filter DataFrame by date range string."""
    if df.empty:
        return df

    df = df.copy()

    if "date" not in df.columns:
        return df

    # Ensure date column is datetime
    if not pd.api.types.is_datetime64_any_dtype(df["date"]):
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # Anchor to the latest available date to avoid timezone drift from local clock.
    reference_date = df["date"].max()
    if pd.isna(reference_date):
        return df

    if date_range == "Last 90 Days":
        cutoff = reference_date - timedelta(days=90)
    elif date_range == "Last 6 Months":
        cutoff = reference_date - timedelta(days=180)
    elif date_range == "Last Year":
        cutoff = reference_date - timedelta(days=365)
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
