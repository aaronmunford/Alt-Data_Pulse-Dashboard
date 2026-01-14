"""
Chart Building Utilities for Dash Dashboard
Bloomberg Terminal Style Plotly Charts with Stock-Chart Interactivity
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

# Standard range selector buttons (like stock charts)
RANGE_SELECTOR_BUTTONS = [
    dict(count=1, label="1M", step="month", stepmode="backward"),
    dict(count=3, label="3M", step="month", stepmode="backward"),
    dict(count=6, label="6M", step="month", stepmode="backward"),
    dict(count=1, label="1Y", step="year", stepmode="backward"),
    dict(step="all", label="All"),
]


def _apply_stock_chart_features(
    fig: go.Figure,
    show_rangeslider: bool = True,
    show_rangeselector: bool = True,
    height: int = 350,
) -> go.Figure:
    """
    Apply TradingView-style features to a Plotly figure.

    Features:
    - Drag to pan across the chart (TradingView-style)
    - Crosshair cursor with spike lines
    - Range selector buttons (1M, 3M, 6M, 1Y, All)
    - Range slider at bottom for dragging time range
    - Scroll wheel zoom

    Args:
        fig: Plotly figure to modify
        show_rangeslider: Whether to show the range slider at bottom
        show_rangeselector: Whether to show range selector buttons
        height: Chart height

    Returns:
        Modified Plotly figure
    """
    xaxis_config = dict(
        showgrid=True,
        gridcolor=COLORS["grid"],
        color=COLORS["text"],
        tickformat="%b %Y",
        showline=True,
        linecolor=COLORS["grid"],
        # TradingView-style crosshair/spike lines
        showspikes=True,
        spikemode="across",
        spikesnap="cursor",
        spikethickness=1,
        spikecolor=COLORS["text_muted"],
        spikedash="solid",
    )

    # Add range selector buttons
    if show_rangeselector:
        xaxis_config["rangeselector"] = dict(
            buttons=RANGE_SELECTOR_BUTTONS,
            bgcolor=COLORS["bg_secondary"],
            activecolor=COLORS["blue"],
            bordercolor=COLORS["grid"],
            borderwidth=1,
            font=dict(color=COLORS["text"], size=10),
            x=0,
            y=1.0,
            yanchor="bottom",
        )

    # Add range slider
    if show_rangeslider:
        xaxis_config["rangeslider"] = dict(
            visible=True,
            bgcolor=COLORS["bg_secondary"],
            bordercolor=COLORS["grid"],
            borderwidth=1,
            thickness=0.08,
        )

    fig.update_xaxes(**xaxis_config)

    # Y-axis crosshair spike line
    fig.update_yaxes(
        showspikes=True,
        spikemode="across",
        spikesnap="cursor",
        spikethickness=1,
        spikecolor=COLORS["text_muted"],
        spikedash="solid",
    )

    # Adjust layout for range slider space
    bottom_margin = 80 if show_rangeslider else 40

    fig.update_layout(
        height=height,
        margin=dict(l=50, r=20, t=40, b=bottom_margin),
        # TradingView-style: drag to pan, scroll to zoom
        dragmode="pan",
        # Crosshair hover mode
        hovermode="x unified",
        # Show spike lines on hover
        spikedistance=-1,
    )

    return fig


def create_traffic_chart(df: pd.DataFrame, date_range: str = "Last Year", fullscreen: bool = False) -> go.Figure:
    """
    Create Foot Traffic time series chart with stock-chart interactivity.

    Args:
        df: DataFrame with columns: date, total_visits, visits_7d_avg
        date_range: Filter for date range (Last 90 Days, Last 6 Months, Last Year, All Available)
        fullscreen: Whether this is for fullscreen modal (larger with more controls)

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
    elif "total_visits" in df_filtered.columns:
        y_values = df_filtered["total_visits"]
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
            line=dict(color=COLORS["green"], width=2),
            fill="tozeroy",
            fillcolor="rgba(63, 185, 80, 0.1)",
            name="Traffic Index",
            hovertemplate="<b>%{x|%b %d, %Y}</b><br>"
            + "Traffic Index: %{y:.1f}<br>"
            + "<extra></extra>",
        )
    )

    # Apply Bloomberg theme
    fig.update_layout(
        yaxis=dict(
            title="Index (Mean = 100)",
            showgrid=True,
            gridcolor=COLORS["grid"],
            color=COLORS["text"],
            zeroline=False,
            side="right",
        ),
        plot_bgcolor=COLORS["bg_primary"],
        paper_bgcolor=COLORS["bg_primary"],
        font=dict(family="IBM Plex Mono", size=11, color=COLORS["text"]),
        hovermode="x unified",
        showlegend=False,
    )

    # Apply stock chart features
    height = 500 if fullscreen else 350
    _apply_stock_chart_features(
        fig,
        show_rangeslider=fullscreen,
        show_rangeselector=True,
        height=height,
    )

    return fig


def create_ticket_chart(df: pd.DataFrame, date_range: str = "Last Year", fullscreen: bool = False) -> go.Figure:
    """
    Create Average Ticket Size time series chart with stock-chart interactivity.

    Args:
        df: DataFrame with columns: date, avg_ticket_size
        date_range: Filter for date range
        fullscreen: Whether this is for fullscreen modal

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
            line=dict(color=COLORS["blue"], width=2),
            fill="tozeroy",
            fillcolor="rgba(88, 166, 255, 0.1)",
            name="Avg Ticket Size",
            hovertemplate="<b>%{x|%b %d, %Y}</b><br>"
            + "Avg Ticket: $%{y:.2f}<br>"
            + "<extra></extra>",
        )
    )

    # Apply Bloomberg theme
    fig.update_layout(
        yaxis=dict(
            title="Average Ticket ($)",
            showgrid=True,
            gridcolor=COLORS["grid"],
            color=COLORS["text"],
            zeroline=False,
            tickprefix="$",
            side="right",
        ),
        plot_bgcolor=COLORS["bg_primary"],
        paper_bgcolor=COLORS["bg_primary"],
        font=dict(family="IBM Plex Mono", size=11, color=COLORS["text"]),
        hovermode="x unified",
        showlegend=False,
    )

    # Apply stock chart features
    height = 500 if fullscreen else 350
    _apply_stock_chart_features(
        fig,
        show_rangeslider=fullscreen,
        show_rangeselector=True,
        height=height,
    )

    return fig


def create_combined_chart(
    df: pd.DataFrame, date_range: str = "Last Year", fullscreen: bool = False
) -> go.Figure:
    """
    Create combined chart with Traffic and Ticket Size on dual y-axes.

    Args:
        df: DataFrame with all required columns
        date_range: Filter for date range
        fullscreen: Whether this is for fullscreen modal

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
                line=dict(color=COLORS["green"], width=2),
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
                line=dict(color=COLORS["blue"], width=2),
                name="Avg Ticket",
                hovertemplate="Ticket: $%{y:.2f}<extra></extra>",
            ),
            secondary_y=True,
        )

    # Update y-axes
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
        plot_bgcolor=COLORS["bg_primary"],
        paper_bgcolor=COLORS["bg_primary"],
        font=dict(family="IBM Plex Mono", size=11, color=COLORS["text"]),
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.0,
            xanchor="right",
            x=1,
            bgcolor="rgba(0,0,0,0)",
        ),
        dragmode="zoom",
    )

    # Apply stock chart features to x-axis
    height = 500 if fullscreen else 400
    fig.update_xaxes(
        showgrid=True,
        gridcolor=COLORS["grid"],
        color=COLORS["text"],
        tickformat="%b %Y",
        showline=True,
        linecolor=COLORS["grid"],
        rangeselector=dict(
            buttons=RANGE_SELECTOR_BUTTONS,
            bgcolor=COLORS["bg_secondary"],
            activecolor=COLORS["blue"],
            bordercolor=COLORS["grid"],
            borderwidth=1,
            font=dict(color=COLORS["text"], size=10),
            x=0,
            y=1.0,
            yanchor="bottom",
        ),
        rangeslider=dict(
            visible=fullscreen,
            bgcolor=COLORS["bg_secondary"],
            bordercolor=COLORS["grid"],
            thickness=0.08,
        ),
    )

    bottom_margin = 80 if fullscreen else 40
    fig.update_layout(
        height=height,
        margin=dict(l=50, r=50, t=50, b=bottom_margin),
    )

    return fig


def create_app_engagement_chart(
    df: pd.DataFrame, date_range: str = "Last Year", fullscreen: bool = False
) -> go.Figure:
    """
    Create App Engagement chart with DAU and Installs on dual y-axes.

    Args:
        df: DataFrame with columns: date, dau, installs (optional)
        date_range: Filter for date range
        fullscreen: Whether this is for fullscreen modal

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
                line=dict(color=COLORS["orange"], width=2),
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

    # Update y-axes
    if has_dau and has_installs:
        fig.update_yaxes(
            title_text="DAU Index",
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
        fig.update_yaxes(
            title_text="DAU Index" if has_dau else "Daily Installs",
            showgrid=True,
            gridcolor=COLORS["grid"],
            color=COLORS["orange"] if has_dau else COLORS["yellow"],
        )

    # Update layout
    fig.update_layout(
        plot_bgcolor=COLORS["bg_primary"],
        paper_bgcolor=COLORS["bg_primary"],
        font=dict(family="IBM Plex Mono", size=11, color=COLORS["text"]),
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.0,
            xanchor="right",
            x=1,
            bgcolor="rgba(0,0,0,0)",
        ),
        dragmode="zoom",
    )

    # Apply stock chart features
    height = 500 if fullscreen else 350
    fig.update_xaxes(
        showgrid=True,
        gridcolor=COLORS["grid"],
        color=COLORS["text"],
        tickformat="%b %Y",
        showline=True,
        linecolor=COLORS["grid"],
        rangeselector=dict(
            buttons=RANGE_SELECTOR_BUTTONS,
            bgcolor=COLORS["bg_secondary"],
            activecolor=COLORS["blue"],
            bordercolor=COLORS["grid"],
            borderwidth=1,
            font=dict(color=COLORS["text"], size=10),
            x=0,
            y=1.0,
            yanchor="bottom",
        ),
        rangeslider=dict(
            visible=fullscreen,
            bgcolor=COLORS["bg_secondary"],
            bordercolor=COLORS["grid"],
            thickness=0.08,
        ),
    )

    bottom_margin = 80 if fullscreen else 40
    fig.update_layout(
        height=height,
        margin=dict(l=50, r=50, t=50, b=bottom_margin),
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


def create_job_openings_donut(
    df: pd.DataFrame, title: str = ""
) -> go.Figure:
    """
    Create a donut chart for job openings breakdown.

    When WRDS/Revelio data is available, this will show breakdown by function.
    For now with LinkUp data, shows inflows vs outflows or placeholder.

    Args:
        df: DataFrame with hiring data
        title: Chart title (optional, usually set in layout)

    Returns:
        Plotly Figure with donut chart
    """
    if df.empty:
        return _create_empty_chart("No job openings data available")

    # Get latest period data
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    latest = df.sort_values("date").iloc[-1]

    # With LinkUp data, show created vs deleted job postings
    created = float(latest.get("created_job_count", 0) or latest.get("inflows", 0) or 0)
    deleted = float(latest.get("deleted_job_count", 0) or latest.get("outflows", 0) or 0)

    if created == 0 and deleted == 0:
        return _create_empty_chart("No job flow data available")

    # Create donut chart
    labels = ["New Openings", "Closed/Filled"]
    values = [created, deleted]
    colors = [COLORS["green"], COLORS["red"]]

    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=0.65,
        marker_colors=colors,
        textinfo="percent",
        textposition="inside",
        textfont=dict(color="#ffffff", size=12, family="IBM Plex Mono"),
        hovertemplate="<b>%{label}</b><br>Count: %{value:,.0f}<br>%{percent}<extra></extra>",
    )])

    # Add center annotation
    total = created + deleted
    fig.add_annotation(
        text=f"<b>{total/1000:.1f}K</b><br><span style='font-size:9px;color:#8b949e'>total</span>",
        x=0.5, y=0.5,
        font=dict(size=18, color=COLORS["text"], family="Space Grotesk"),
        showarrow=False,
    )

    fig.update_layout(
        plot_bgcolor=COLORS["bg_primary"],
        paper_bgcolor=COLORS["bg_primary"],
        font=dict(family="IBM Plex Mono", size=11, color=COLORS["text"]),
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.1,
            xanchor="center",
            x=0.5,
            bgcolor="rgba(0,0,0,0)",
            font=dict(size=10),
        ),
        margin=dict(l=10, r=10, t=10, b=40),
        height=260,
    )

    return fig


def create_employee_count_chart(
    df: pd.DataFrame, date_range: str = "Last Year", fullscreen: bool = False
) -> go.Figure:
    """
    Create Employee Count area chart with stock-chart interactivity.

    Args:
        df: DataFrame with hiring data
        date_range: Filter for date range
        fullscreen: Whether this is for fullscreen modal

    Returns:
        Plotly Figure with area chart
    """
    if df.empty:
        return _create_empty_chart("No employee count data available")

    df_filtered = _filter_by_date_range(df, date_range)

    if df_filtered.empty:
        return _create_empty_chart("No data in selected range")

    df_filtered = df_filtered.copy()

    # Use headcount or unique_active_job_count
    y_col = "headcount" if "headcount" in df_filtered.columns else "unique_active_job_count"

    if y_col not in df_filtered.columns or df_filtered[y_col].sum() == 0:
        return _create_empty_chart("No employee/headcount data available")

    fig = go.Figure()

    # Add area chart with markers (like LinkedIn)
    fig.add_trace(
        go.Scatter(
            x=df_filtered["date"],
            y=df_filtered[y_col],
            mode="lines+markers",
            fill="tozeroy",
            line=dict(color=COLORS["blue"], width=2),
            marker=dict(size=5, color=COLORS["bg_primary"], line=dict(color=COLORS["blue"], width=2)),
            fillcolor="rgba(88, 166, 255, 0.15)",
            name="Job Postings",
            hovertemplate="<b>%{x|%b %Y}</b><br>Count: %{y:,.0f}<extra></extra>",
        )
    )

    fig.update_layout(
        yaxis=dict(
            title="",
            showgrid=True,
            gridcolor=COLORS["grid"],
            color=COLORS["text"],
            tickformat=",",
            side="right",
        ),
        plot_bgcolor=COLORS["bg_primary"],
        paper_bgcolor=COLORS["bg_primary"],
        font=dict(family="IBM Plex Mono", size=11, color=COLORS["text"]),
        hovermode="x unified",
        showlegend=False,
        dragmode="zoom",
    )

    # Apply stock chart features
    height = 450 if fullscreen else 220
    _apply_stock_chart_features(
        fig,
        show_rangeslider=fullscreen,
        show_rangeselector=True,
        height=height,
    )

    return fig


def create_hiring_flow_chart(
    df: pd.DataFrame, date_range: str = "Last Year", fullscreen: bool = False
) -> go.Figure:
    """
    Create Hiring Flow chart showing inflows (new hires) vs outflows (departures).

    Args:
        df: DataFrame with hiring data (needs inflows, outflows columns)
        date_range: Filter for date range
        fullscreen: Whether this is for fullscreen modal

    Returns:
        Plotly Figure with stacked area chart
    """
    if df.empty:
        return _create_empty_chart("No hiring flow data available")

    df_filtered = _filter_by_date_range(df, date_range)

    if df_filtered.empty:
        return _create_empty_chart("No data in selected range")

    has_inflows = "inflows" in df_filtered.columns
    has_outflows = "outflows" in df_filtered.columns

    if not has_inflows and not has_outflows:
        return _create_empty_chart("No inflow/outflow data available")

    df_filtered = df_filtered.copy()

    fig = go.Figure()

    # Inflows (positive - new job postings/hires)
    if has_inflows:
        fig.add_trace(
            go.Scatter(
                x=df_filtered["date"],
                y=df_filtered["inflows"],
                mode="lines",
                fill="tozeroy",
                line=dict(color=COLORS["green"], width=1.5),
                fillcolor="rgba(63, 185, 80, 0.3)",
                name="New Postings",
                hovertemplate="<b>%{x|%b %Y}</b><br>New: %{y:,.0f}<extra></extra>",
            )
        )

    # Outflows
    if has_outflows:
        fig.add_trace(
            go.Scatter(
                x=df_filtered["date"],
                y=df_filtered["outflows"],
                mode="lines",
                fill="tozeroy",
                line=dict(color=COLORS["red"], width=1.5),
                fillcolor="rgba(248, 81, 73, 0.3)",
                name="Closed/Filled",
                hovertemplate="<b>%{x|%b %Y}</b><br>Closed: %{y:,.0f}<extra></extra>",
            )
        )

    fig.update_layout(
        yaxis=dict(
            title="",
            showgrid=True,
            gridcolor=COLORS["grid"],
            color=COLORS["text"],
            tickformat=",",
        ),
        plot_bgcolor=COLORS["bg_primary"],
        paper_bgcolor=COLORS["bg_primary"],
        font=dict(family="IBM Plex Mono", size=11, color=COLORS["text"]),
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.2,
            xanchor="center",
            x=0.5,
            bgcolor="rgba(0,0,0,0)",
            font=dict(size=10),
        ),
        dragmode="zoom",
    )

    # Apply stock chart features
    height = 450 if fullscreen else 250
    _apply_stock_chart_features(
        fig,
        show_rangeslider=fullscreen,
        show_rangeselector=True,
        height=height,
    )

    return fig


def create_hiring_velocity_chart(
    df: pd.DataFrame, date_range: str = "Last Year", fullscreen: bool = False
) -> go.Figure:
    """
    Create Hiring Velocity chart (month-over-month % change).

    Args:
        df: DataFrame with hiring_velocity column
        date_range: Filter for date range
        fullscreen: Whether this is for fullscreen modal

    Returns:
        Plotly Figure with bar chart
    """
    if df.empty:
        return _create_empty_chart("No hiring velocity data available")

    df_filtered = _filter_by_date_range(df, date_range)

    if df_filtered.empty or "hiring_velocity" not in df_filtered.columns:
        return _create_empty_chart("No velocity data in selected range")

    df_filtered = df_filtered.copy()

    # Color bars based on positive/negative
    colors = [COLORS["green"] if v >= 0 else COLORS["red"]
              for v in df_filtered["hiring_velocity"]]

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=df_filtered["date"],
            y=df_filtered["hiring_velocity"],
            marker_color=colors,
            name="Hiring Velocity",
            hovertemplate="<b>%{x|%b %Y}</b><br>Velocity: %{y:+.1f}%<extra></extra>",
        )
    )

    # Add zero line
    fig.add_hline(y=0, line_dash="solid", line_color=COLORS["text_muted"], line_width=1)

    fig.update_layout(
        yaxis=dict(
            title="",
            showgrid=True,
            gridcolor=COLORS["grid"],
            color=COLORS["text"],
            ticksuffix="%",
        ),
        plot_bgcolor=COLORS["bg_primary"],
        paper_bgcolor=COLORS["bg_primary"],
        font=dict(family="IBM Plex Mono", size=11, color=COLORS["text"]),
        hovermode="x unified",
        showlegend=False,
        dragmode="zoom",
    )

    # Apply stock chart features
    height = 450 if fullscreen else 250
    _apply_stock_chart_features(
        fig,
        show_rangeslider=fullscreen,
        show_rangeselector=True,
        height=height,
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
