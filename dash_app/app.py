"""
Alt-Data Pulse Dashboard - Plotly Dash Version
Phase 2H: Production-Ready Bloomberg-Style Dashboard
"""

import dash
from dash import html, dcc, callback, Input, Output, State, ALL, ctx
import dash_bootstrap_components as dbc
import pandas as pd
import json
import uuid
from datetime import datetime

# Import data utilities
from utils import (
    load_brand_signal,
    load_brand_trend_data,
    load_brand_news,
    load_story_content,
    get_available_brands,
    format_currency,
    format_delta,
    get_delta_color,
    get_signal_color,
)

# Import chart builders
from charts import create_traffic_chart, create_ticket_chart, create_combined_chart

# Import news feed component
from components.news_feed import (
    create_news_feed_panel,
    create_news_unavailable_message,
    create_story_modal,
)

# Initialize Dash app with external stylesheets
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    suppress_callback_exceptions=True,
    title="Alt-Data Pulse Dashboard",
)

# Server for deployment
server = app.server

# ============================================================================
# Workspace Tabs
# ============================================================================

DEFAULT_WORKSPACE_TITLE = "Start Page"


def _default_workspace():
    return {
        "id": "start-page",
        "title": DEFAULT_WORKSPACE_TITLE,
        "template": "markets",
        "state": {},
    }


def _new_workspace():
    return {
        "id": f"ws-{uuid.uuid4().hex[:8]}",
        "title": DEFAULT_WORKSPACE_TITLE,
        "template": "markets",
        "state": {},
    }


def _normalize_workspace_state(data):
    if not data or not isinstance(data, dict):
        workspace = _default_workspace()
        return {"workspaces": [workspace], "active_id": workspace["id"]}

    raw_workspaces = [ws for ws in data.get("workspaces", []) if isinstance(ws, dict)]
    if not raw_workspaces:
        workspace = _default_workspace()
        return {"workspaces": [workspace], "active_id": workspace["id"]}

    normalized = []
    for ws in raw_workspaces:
        ws_id = ws.get("id") or f"ws-{uuid.uuid4().hex[:8]}"
        template = ws.get("template") or "markets"
        if template == "start":
            template = "markets"
        normalized.append(
            {
                "id": ws_id,
                "title": ws.get("title") or DEFAULT_WORKSPACE_TITLE,
                "template": template,
                "state": ws.get("state") or {},
            }
        )

    active_id = data.get("active_id")
    if active_id not in {ws["id"] for ws in normalized}:
        active_id = normalized[0]["id"]

    return {"workspaces": normalized, "active_id": active_id}


def _start_page_layout(workspaces, active_id):
    workspace_items = []
    for ws in workspaces:
        is_active = ws["id"] == active_id
        workspace_items.append(
            html.Div(
                [
                    html.Span(ws["title"], className="start-page-workspace-title"),
                    html.Span(
                        "ACTIVE" if is_active else "OPEN",
                        className="start-page-workspace-tag"
                        + (" start-page-workspace-tag-active" if is_active else ""),
                    ),
                ],
                className="start-page-workspace-item",
            )
        )

    return dbc.Container(
        [
            dbc.Row(
                [
                    dbc.Col(
                        html.Div(
                            [
                                html.H2(
                                    "Start Page",
                                    className="start-page-title",
                                ),
                                html.P(
                                    "Open new workspaces, jump to saved layouts, or type a command.",
                                    className="start-page-subtitle",
                                ),
                            ],
                            className="start-page-hero",
                        ),
                        width=12,
                    )
                ]
            ),
            dbc.Row(
                [
                    dbc.Col(
                        dcc.Input(
                            type="text",
                            placeholder="Type a command or ticker (e.g., SBUX US)",
                            className="start-page-input",
                        ),
                        width=12,
                    )
                ],
                className="start-page-command-row",
            ),
            dbc.Row(
                [
                    dbc.Col(
                        html.Div(
                            [
                                html.H3("Workspaces", className="start-page-card-title"),
                                html.Div(
                                    workspace_items,
                                    className="start-page-workspace-list",
                                ),
                            ],
                            className="dash-card start-page-card",
                        ),
                        md=4,
                    ),
                    dbc.Col(
                        html.Div(
                            [
                                html.H3("Quick Launch", className="start-page-card-title"),
                                html.P(
                                    "Markets dashboard, news tape, and charting tools.",
                                    className="start-page-card-body",
                                ),
                                html.P(
                                    "Add a workspace with the plus button to open a new start page.",
                                    className="start-page-card-body",
                                ),
                            ],
                            className="dash-card start-page-card",
                        ),
                        md=4,
                    ),
                    dbc.Col(
                        html.Div(
                            [
                                html.H3("Tips", className="start-page-card-title"),
                                html.P(
                                    "Keep multiple layouts open for faster switching.",
                                    className="start-page-card-body",
                                ),
                                html.P(
                                    "Use short labels to keep tabs compact.",
                                    className="start-page-card-body",
                                ),
                            ],
                            className="dash-card start-page-card",
                        ),
                        md=4,
                    ),
                ],
                className="start-page-grid",
            ),
        ],
        fluid=True,
        style={"maxWidth": "1400px"},
        className="workspace-pane",
    )

# ============================================================================
# Layout
# ============================================================================

markets_layout = dbc.Container(
    [
        # Auto-refresh interval component (5 minutes)
        dcc.Interval(
            id="interval-component",
            interval=5 * 60 * 1000,  # 5 minutes in milliseconds
            n_intervals=0,
        ),
        # Store component for manual refresh trigger
        dcc.Store(id="refresh-trigger", data=0),
        # Download component for CSV export
        dcc.Download(id="download-dataframe-csv"),
        # News-specific interval (matches main refresh - 5 minutes)
        dcc.Interval(
            id="news-interval-component",
            interval=5 * 60 * 1000,  # 5 minutes in milliseconds
            n_intervals=0,
        ),
        # Store for news data
        dcc.Store(id="news-data-store", data=[]),
        # Story modal for displaying full news articles
        create_story_modal(),
        # Header Section
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.H1(
                            "Alt-Data Pulse Dashboard",
                            style={
                                "fontFamily": "'Space Grotesk', sans-serif",
                                "fontWeight": "600",
                                "marginTop": "2rem",
                                "marginBottom": "0.5rem",
                            },
                        ),
                        html.P(
                            "Hedge Fund Grade Quantitative Signal Platform",
                            style={
                                "fontFamily": "'IBM Plex Mono', monospace",
                                "fontSize": "0.875rem",
                                "color": "#8b949e",
                                "marginBottom": "0rem",
                            },
                        ),
                        html.Div(
                            id="last-updated-display",
                            style={
                                "fontFamily": "'IBM Plex Mono', monospace",
                                "fontSize": "0.75rem",
                                "color": "#6e7681",
                                "marginBottom": "1rem",
                            },
                        ),
                    ],
                    md=9,
                ),
                dbc.Col(
                    [
                        html.Div(
                            [
                                dbc.Row(
                                    [
                                        dbc.Col(
                                            dbc.Button(
                                                "↻ Refresh",
                                                id="refresh-button",
                                                color="secondary",
                                                size="sm",
                                                style={
                                                    "marginTop": "2rem",
                                                    "marginBottom": "0.5rem",
                                                    "width": "100%",
                                                },
                                            ),
                                            width=6,
                                        ),
                                        dbc.Col(
                                            dbc.Button(
                                                "⬇ Export CSV",
                                                id="export-csv-button",
                                                color="primary",
                                                size="sm",
                                                style={
                                                    "marginTop": "2rem",
                                                    "marginBottom": "0.5rem",
                                                    "width": "100%",
                                                },
                                            ),
                                            width=6,
                                        ),
                                    ]
                                ),
                                dcc.Dropdown(
                                    id="refresh-interval-selector",
                                    options=[
                                        {"label": "Auto-refresh: Off", "value": 0},
                                        {"label": "Auto-refresh: 1 min", "value": 1},
                                        {"label": "Auto-refresh: 5 min", "value": 5},
                                        {"label": "Auto-refresh: 15 min", "value": 15},
                                    ],
                                    value=5,  # Default 5 minutes
                                    clearable=False,
                                    style={
                                        "fontSize": "0.75rem",
                                    },
                                ),
                            ]
                        )
                    ],
                    md=3,
                ),
            ]
        ),
        # Controls Row: Brand Selector + Date Range
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.Label(
                            "Select Brand:",
                            style={
                                "fontFamily": "'Space Grotesk', sans-serif",
                                "fontSize": "0.875rem",
                                "fontWeight": "600",
                                "marginBottom": "0.5rem",
                                "display": "block",
                            },
                        ),
                        dcc.Dropdown(
                            id="brand-selector",
                            options=[
                                {"label": brand, "value": brand}
                                for brand in get_available_brands()
                            ],
                            value="STARBUCKS",  # Default selection
                            clearable=False,
                            style={
                                "backgroundColor": "#161b33",
                                "color": "#e6edf3",
                            },
                        ),
                    ],
                    md=4,
                ),
                dbc.Col(
                    [
                        html.Label(
                            "Time Range:",
                            style={
                                "fontFamily": "'Space Grotesk', sans-serif",
                                "fontSize": "0.875rem",
                                "fontWeight": "600",
                                "marginBottom": "0.5rem",
                                "display": "block",
                            },
                        ),
                        dcc.Dropdown(
                            id="date-range-selector",
                            options=[
                                {"label": "Last 90 Days", "value": "Last 90 Days"},
                                {"label": "Last 6 Months", "value": "Last 6 Months"},
                                {"label": "Last Year", "value": "Last Year"},
                                {"label": "All Available", "value": "All Available"},
                            ],
                            value="Last Year",  # Default selection
                            clearable=False,
                            style={
                                "backgroundColor": "#161b33",
                                "color": "#e6edf3",
                            },
                        ),
                    ],
                    md=4,
                ),
            ],
            style={"marginBottom": "2rem"},
        ),
        # Status Badge
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.Div(
                            "🚀 Production-Ready Dashboard (Phase 2H Complete)",
                            style={
                                "backgroundColor": "#1a1f3a",
                                "border": "1px solid #3fb950",
                                "borderRadius": "6px",
                                "padding": "0.75rem 1.25rem",
                                "fontSize": "0.875rem",
                                "color": "#3fb950",
                                "fontWeight": "600",
                                "marginBottom": "2rem",
                                "display": "inline-block",
                            },
                        )
                    ],
                    width=12,
                )
            ]
        ),
        # KPI Cards Row (Dynamic)
        dbc.Row(
            [
                dbc.Col(
                    html.Div(id="kpi-revenue-proxy"),
                    md=3,
                ),
                dbc.Col(
                    html.Div(id="kpi-consensus"),
                    md=3,
                ),
                dbc.Col(
                    html.Div(id="kpi-delta"),
                    md=3,
                ),
                dbc.Col(
                    html.Div(id="kpi-signal"),
                    md=3,
                ),
            ],
            style={"marginBottom": "2rem"},
        ),
        # Charts Section
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.Div(
                            [
                                dcc.Graph(
                                    id="traffic-chart",
                                    config={
                                        "displayModeBar": True,
                                        "displaylogo": False,
                                        "modeBarButtonsToRemove": [
                                            "select2d",
                                            "lasso2d",
                                        ],
                                    },
                                )
                            ],
                            className="dash-card",
                            style={"padding": "1rem"},
                        )
                    ],
                    md=6,
                ),
                dbc.Col(
                    [
                        html.Div(
                            [
                                dcc.Graph(
                                    id="ticket-chart",
                                    config={
                                        "displayModeBar": True,
                                        "displaylogo": False,
                                        "modeBarButtonsToRemove": [
                                            "select2d",
                                            "lasso2d",
                                        ],
                                    },
                                )
                            ],
                            className="dash-card",
                            style={"padding": "1rem"},
                        )
                    ],
                    md=6,
                ),
            ],
            style={"marginBottom": "2rem"},
        ),
        # Combined Chart + News Panel Section
        dbc.Row(
            [
                # Combined Chart (9 columns)
                dbc.Col(
                    [
                        html.Div(
                            [
                                dcc.Graph(
                                    id="combined-chart",
                                    config={
                                        "displayModeBar": True,
                                        "displaylogo": False,
                                        "modeBarButtonsToRemove": [
                                            "select2d",
                                            "lasso2d",
                                        ],
                                    },
                                )
                            ],
                            className="dash-card",
                            style={"padding": "1rem"},
                        )
                    ],
                    md=9,
                ),
                # News Feed Panel (3 columns)
                dbc.Col(
                    [
                        html.Div(
                            id="news-feed-container",
                            children=create_news_unavailable_message(),
                        )
                    ],
                    md=3,
                ),
            ],
            style={"marginBottom": "2rem"},
        ),
        # Debug Section Toggle
        dbc.Row(
            [
                dbc.Col(
                    [
                        dbc.Button(
                            "▼ Show Debug Data",
                            id="collapse-button",
                            color="link",
                            size="sm",
                            style={
                                "color": "#8b949e",
                                "fontSize": "0.875rem",
                                "marginBottom": "1rem",
                            },
                        ),
                    ],
                    width=12,
                )
            ]
        ),
        # Collapsible Debug Sections
        dbc.Collapse(
            [
                # Raw Data Display Section
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                html.Div(
                                    [
                                        html.H3(
                                            "Raw Signal Data (Debug Output)",
                                            className="font-sans",
                                            style={"marginBottom": "1rem"},
                                        ),
                                        html.Pre(
                                            id="raw-signal-data",
                                            style={
                                                "backgroundColor": "#0a0e27",
                                                "border": "1px solid #30363d",
                                                "borderRadius": "6px",
                                                "padding": "1.5rem",
                                                "fontSize": "0.875rem",
                                                "color": "#e6edf3",
                                                "fontFamily": "'IBM Plex Mono', monospace",
                                                "overflow": "auto",
                                                "maxHeight": "400px",
                                            },
                                        ),
                                    ],
                                    className="dash-card",
                                )
                            ],
                            width=12,
                        )
                    ]
                ),
                # Trend Data Preview
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                html.Div(
                                    [
                                        html.H3(
                                            "Trend Data Preview (First 10 Rows)",
                                            className="font-sans",
                                            style={"marginBottom": "1rem"},
                                        ),
                                        html.Pre(
                                            id="trend-data-preview",
                                            style={
                                                "backgroundColor": "#0a0e27",
                                                "border": "1px solid #30363d",
                                                "borderRadius": "6px",
                                                "padding": "1.5rem",
                                                "fontSize": "0.75rem",
                                                "color": "#8b949e",
                                                "fontFamily": "'IBM Plex Mono', monospace",
                                                "overflow": "auto",
                                                "maxHeight": "500px",
                                            },
                                        ),
                                    ],
                                    className="dash-card",
                                )
                            ],
                            width=12,
                        )
                    ]
                ),
            ],
            id="collapse-debug",
            is_open=False,
        ),
        # Footer
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.Hr(style={"borderColor": "#30363d", "margin": "2rem 0"}),
                        html.P(
                            [
                                "Built with ",
                                html.A(
                                    "Plotly Dash",
                                    href="https://dash.plotly.com/",
                                    target="_blank",
                                    style={"color": "#58a6ff"},
                                ),
                                " • Bloomberg Terminal Style • All Phases Complete",
                            ],
                            style={
                                "fontSize": "0.75rem",
                                "color": "#6e7681",
                                "textAlign": "center",
                                "marginBottom": "2rem",
                            },
                        ),
                    ],
                    width=12,
                )
            ]
        ),
    ],
    fluid=True,
    style={"maxWidth": "1400px"},
)

DEFAULT_WORKSPACE_STATE = {
    "workspaces": [_default_workspace()],
    "active_id": "start-page",
}


def _render_workspace_tabs(workspaces, active_id):
    tabs = []
    for ws in workspaces:
        is_active = ws["id"] == active_id
        tabs.append(
            html.Div(
                [
                    html.Span(ws["title"], className="workspace-tab-title"),
                    html.Button(
                        "x",
                        id={"type": "workspace-close", "id": ws["id"]},
                        className="workspace-tab-close",
                        n_clicks=0,
                        type="button",
                    ),
                ],
                id={"type": "workspace-tab", "id": ws["id"]},
                className="workspace-tab"
                + (" workspace-tab-active" if is_active else ""),
                n_clicks=0,
                role="button",
                tabIndex=0,
            )
        )
    return tabs


def _render_workspace_content(workspace, state):
    template = workspace.get("template", "markets")
    if template in ("markets", "start"):
        return markets_layout
    return _start_page_layout(state["workspaces"], state["active_id"])


app.layout = html.Div(
    [
        dcc.Store(
            id="workspace-store",
            storage_type="local",
            data=DEFAULT_WORKSPACE_STATE,
        ),
        html.Div(
            [
                dbc.Container(
                    html.Div(
                        [
                            html.Div(
                                id="workspace-tabs",
                                className="workspace-tabs",
                            ),
                            html.Button(
                                "+",
                                id="workspace-add",
                                className="workspace-tab-add",
                                n_clicks=0,
                                type="button",
                            ),
                        ],
                        className="workspace-tabs-inner",
                    ),
                    fluid=True,
                    style={"maxWidth": "1400px"},
                    className="workspace-tabs-container",
                )
            ],
            className="workspace-tab-bar",
        ),
        html.Div(
            id="workspace-content",
            className="workspace-content",
        ),
    ]
)

# ============================================================================
# Callbacks
# ============================================================================

@callback(
    Output("workspace-store", "data"),
    [
        Input("workspace-add", "n_clicks"),
        Input({"type": "workspace-tab", "id": ALL}, "n_clicks"),
        Input({"type": "workspace-close", "id": ALL}, "n_clicks"),
    ],
    State("workspace-store", "data"),
    prevent_initial_call=True,
)
def update_workspace_store(add_clicks, tab_clicks, close_clicks, store_data):
    """Handle workspace tab add/select/close actions."""
    state = _normalize_workspace_state(store_data)

    action = None
    select_id = None
    close_id = None

    for trigger in ctx.triggered or []:
        if trigger.get("value") in (None, 0):
            continue
        prop_id = trigger["prop_id"].split(".")[0]
        if prop_id == "workspace-add":
            action = "add"
            break
        try:
            id_dict = json.loads(prop_id)
        except ValueError:
            continue
        if id_dict.get("type") == "workspace-close":
            action = "close"
            close_id = id_dict.get("id")
        elif id_dict.get("type") == "workspace-tab" and action != "close":
            action = "select"
            select_id = id_dict.get("id")

    if action == "add":
        new_workspace = _new_workspace()
        state["workspaces"] = state["workspaces"] + [new_workspace]
        state["active_id"] = new_workspace["id"]
        return state

    if action == "close" and close_id:
        workspaces = state["workspaces"]
        close_index = next(
            (idx for idx, ws in enumerate(workspaces) if ws["id"] == close_id),
            None,
        )
        workspaces = [ws for ws in workspaces if ws["id"] != close_id]
        if not workspaces:
            workspace = _default_workspace()
            state["workspaces"] = [workspace]
            state["active_id"] = workspace["id"]
            return state
        if state["active_id"] == close_id:
            new_index = min(close_index or 0, len(workspaces) - 1)
            state["active_id"] = workspaces[new_index]["id"]
        state["workspaces"] = workspaces
        return state

    if action == "select" and select_id:
        state["active_id"] = select_id
        return state

    return state


@callback(
    Output("workspace-tabs", "children"),
    Output("workspace-content", "children"),
    Input("workspace-store", "data"),
)
def render_workspace_ui(store_data):
    """Render workspace tabs and the active workspace content."""
    state = _normalize_workspace_state(store_data)
    tabs = _render_workspace_tabs(state["workspaces"], state["active_id"])
    active_workspace = next(
        (ws for ws in state["workspaces"] if ws["id"] == state["active_id"]),
        state["workspaces"][0],
    )
    content = _render_workspace_content(active_workspace, state)
    return tabs, content

@callback(
    [
        Output("kpi-revenue-proxy", "children"),
        Output("kpi-consensus", "children"),
        Output("kpi-delta", "children"),
        Output("kpi-signal", "children"),
        Output("traffic-chart", "figure"),
        Output("ticket-chart", "figure"),
        Output("combined-chart", "figure"),
        Output("raw-signal-data", "children"),
        Output("trend-data-preview", "children"),
    ],
    [
        Input("brand-selector", "value"),
        Input("date-range-selector", "value"),
        Input("refresh-trigger", "data"),
        Input("interval-component", "n_intervals"),
    ],
)
def update_dashboard(brand: str, date_range: str, refresh_trigger, n_intervals):
    """Update all dashboard components when brand, date range, or refresh triggers change."""

    # Load signal data
    signal = load_brand_signal(brand)

    # Handle errors
    if "error" in signal:
        error_msg = f"Error loading data for {brand}:\n{signal['error']}"
        empty_kpi = html.Div("—", className="kpi-card")
        empty_fig = create_traffic_chart(pd.DataFrame())
        return (
            empty_kpi,
            empty_kpi,
            empty_kpi,
            empty_kpi,
            empty_fig,
            empty_fig,
            empty_fig,
            error_msg,
            "No trend data available",
        )

    # Build KPI Cards
    kpi_revenue = html.Div(
        [
            html.Div(
                format_currency(signal["predicted_revenue"]),
                className="kpi-value",
                style={"color": "#58a6ff"},
            ),
            html.Div("Revenue Proxy", className="kpi-label"),
            html.Div(
                signal["quarter"],
                style={
                    "fontSize": "0.75rem",
                    "color": "#6e7681",
                    "marginTop": "0.5rem",
                },
            ),
        ],
        className="kpi-card",
    )

    kpi_consensus = html.Div(
        [
            html.Div(
                format_currency(signal["wall_street_consensus"]),
                className="kpi-value",
                style={"color": "#8b949e"},
            ),
            html.Div("Wall St. Consensus", className="kpi-label"),
            html.Div(
                signal.get("consensus_source", "N/A"),
                style={
                    "fontSize": "0.75rem",
                    "color": "#6e7681",
                    "marginTop": "0.5rem",
                },
            ),
        ],
        className="kpi-card",
    )

    delta_color = get_delta_color(signal["delta_pct"])
    kpi_delta = html.Div(
        [
            html.Div(
                format_delta(signal["delta_pct"]),
                className="kpi-value",
                style={"color": delta_color},
            ),
            html.Div("Delta vs Consensus", className="kpi-label"),
            html.Div(
                signal["delta_direction"],
                style={
                    "fontSize": "0.875rem",
                    "color": delta_color,
                    "marginTop": "0.5rem",
                    "fontWeight": "700",
                },
            ),
        ],
        className="kpi-card",
    )

    signal_color = get_signal_color(signal["signal_strength"])
    kpi_signal = html.Div(
        [
            html.Div(
                f"{signal['correlation']:.3f}",
                className="kpi-value",
                style={"color": signal_color},
            ),
            html.Div("Signal Strength", className="kpi-label"),
            html.Div(
                f"{signal['signal_strength']} (R²)",
                style={
                    "fontSize": "0.75rem",
                    "color": "#6e7681",
                    "marginTop": "0.5rem",
                },
            ),
        ],
        className="kpi-card",
    )

    # Format raw signal data as JSON
    raw_signal_text = json.dumps(signal, indent=2, default=str)

    # Load trend data
    trend_df = load_brand_trend_data(brand)

    # Create charts
    traffic_fig = create_traffic_chart(trend_df, date_range)
    ticket_fig = create_ticket_chart(trend_df, date_range)
    combined_fig = create_combined_chart(trend_df, date_range)

    if trend_df.empty:
        trend_preview_text = "No trend data available"
    else:
        # Show first 10 rows with key columns
        cols_to_show = [
            col
            for col in [
                "date",
                "spend",
                "transactions",
                "avg_ticket_size",
                "spend_7d_avg",
                "total_visits",
                "visits_7d_avg",
            ]
            if col in trend_df.columns
        ]
        preview_df = trend_df[cols_to_show].head(10)
        trend_preview_text = preview_df.to_string()

        # Add summary stats
        trend_preview_text += f"\n\nTotal Rows: {len(trend_df)}"
        trend_preview_text += f"\nDate Range: {trend_df['date'].min()} to {trend_df['date'].max()}"

    return (
        kpi_revenue,
        kpi_consensus,
        kpi_delta,
        kpi_signal,
        traffic_fig,
        ticket_fig,
        combined_fig,
        raw_signal_text,
        trend_preview_text,
    )


@callback(
    Output("interval-component", "interval"),
    Input("refresh-interval-selector", "value"),
)
def update_refresh_interval(minutes: int):
    """Update the auto-refresh interval based on user selection."""
    if minutes == 0:
        # Disable by setting to a very long interval (~1 year).
        return 365 * 24 * 60 * 60 * 1000
    return minutes * 60 * 1000


@callback(
    Output("refresh-trigger", "data"),
    Input("refresh-button", "n_clicks"),
    State("refresh-trigger", "data"),
    prevent_initial_call=True,
)
def manual_refresh(n_clicks, current_value):
    """Increment refresh trigger when button is clicked."""
    if n_clicks is None:
        return current_value
    return current_value + 1


@callback(
    Output("last-updated-display", "children"),
    [
        Input("brand-selector", "value"),
        Input("refresh-trigger", "data"),
        Input("interval-component", "n_intervals"),
    ],
)
def update_timestamp(brand, refresh_trigger, n_intervals):
    """Update the last updated timestamp."""
    now = datetime.now()
    return f"Last updated: {now.strftime('%Y-%m-%d %H:%M:%S')}"


@callback(
    Output("download-dataframe-csv", "data"),
    Input("export-csv-button", "n_clicks"),
    [
        State("brand-selector", "value"),
        State("date-range-selector", "value"),
    ],
    prevent_initial_call=True,
)
def export_csv(n_clicks, brand, date_range):
    """Export trend data as CSV when button is clicked."""
    if n_clicks is None:
        return None

    # Load trend data
    trend_df = load_brand_trend_data(brand)

    if trend_df.empty:
        return None

    # Filter by date range
    from charts import _filter_by_date_range

    trend_df = _filter_by_date_range(trend_df, date_range)

    # Format filename with brand and timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{brand}_{date_range.replace(' ', '_')}_{timestamp}.csv"

    return dcc.send_data_frame(trend_df.to_csv, filename, index=False)


@callback(
    [
        Output("collapse-debug", "is_open"),
        Output("collapse-button", "children"),
    ],
    Input("collapse-button", "n_clicks"),
    State("collapse-debug", "is_open"),
)
def toggle_collapse(n_clicks, is_open):
    """Toggle the debug sections collapse."""
    if n_clicks:
        is_open = not is_open
    button_text = "▲ Hide Debug Data" if is_open else "▼ Show Debug Data"
    return is_open, button_text


# ============================================================================
# News Feed Callback
# ============================================================================


@callback(
    [
        Output("news-feed-container", "children"),
        Output("news-data-store", "data"),
    ],
    [
        Input("brand-selector", "value"),
        Input("news-interval-component", "n_intervals"),
        Input("refresh-trigger", "data"),
    ],
)
def update_news_feed(brand: str, n_intervals: int, refresh_trigger):
    """
    Update news feed when brand changes or news interval fires.

    Args:
        brand: Currently selected brand
        n_intervals: News interval tick count
        refresh_trigger: Manual refresh trigger

    Returns:
        Tuple of (news panel component, news data for store)
    """
    try:
        headlines = load_brand_news(brand, count=15)

        if not headlines:
            return create_news_unavailable_message(), []

        # Check if any headlines are from stale cache
        any_stale = any(h.get("is_stale", False) for h in headlines)

        return create_news_feed_panel(headlines, show_stale_indicator=any_stale), headlines

    except Exception as e:
        print(f"Error updating news feed: {e}")
        return create_news_unavailable_message(), []


# ============================================================================
# Story Modal Callback
# ============================================================================


@callback(
    [
        Output("story-modal", "is_open"),
        Output("story-modal-title", "children"),
        Output("story-modal-meta", "children"),
        Output("story-modal-content", "children"),
    ],
    [
        Input({"type": "news-headline", "index": ALL, "story_id": ALL}, "n_clicks"),
    ],
    [
        State("news-data-store", "data"),
        State("story-modal", "is_open"),
    ],
    prevent_initial_call=True,
)
def open_story_modal(n_clicks_list, news_data, is_open):
    """
    Open story modal when a headline is clicked.

    Args:
        n_clicks_list: List of click counts for all headlines
        news_data: Stored news data from the feed
        is_open: Current modal state

    Returns:
        Tuple of (is_open, title, meta, content)
    """
    # Check if any headline was actually clicked
    if not any(n_clicks_list):
        return False, "", "", ""

    # Find which headline was clicked
    triggered = ctx.triggered_id
    if not triggered:
        return False, "", "", ""

    clicked_index = triggered.get("index")
    story_id = triggered.get("story_id")

    if clicked_index is None or not news_data:
        return False, "", "", ""

    # Get the headline data
    if clicked_index >= len(news_data):
        return False, "", "", ""

    headline_data = news_data[clicked_index]
    headline_text = headline_data.get("headline", "")
    source = headline_data.get("source", "Unknown")
    timestamp = headline_data.get("timestamp", "")

    # Format timestamp for display
    from utils import format_news_timestamp
    time_display = format_news_timestamp(timestamp)

    # Fetch the full story content
    story_content = load_story_content(story_id)

    if story_content:
        content_text = story_content
    else:
        content_text = "Full story content is not available. This may be due to LSEG connection issues or the story is no longer accessible."

    # Build meta info
    meta_text = f"{source} • {time_display}"
    if headline_data.get("is_translated"):
        meta_text += " • Translated"

    return True, headline_text, meta_text, content_text


# ============================================================================
# Run Server
# ============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("🚀 Alt-Data Pulse Dashboard - PRODUCTION READY")
    print("=" * 70)
    print("\n✓ Phase 2H Complete: All features implemented!")
    print("\n📊 Dashboard URL: http://localhost:8050")
    print("\nFeatures:")
    print("  • Bloomberg Terminal dark theme")
    print("  • Interactive Plotly time series charts")
    print("  • Real-time data from RevenuePredictor")
    print("  • Live news feed from LSEG/Refinitiv (brand-filtered)")
    print("  • Auto-refresh (1, 5, 15 min or off)")
    print("  • CSV export with date filtering")
    print("  • Brand selector (STARBUCKS, MCDONALD'S, CHIPOTLE)")
    print("  • KPI cards: Revenue, Consensus, Delta, Signal Strength")
    print("\n⏹  Press Ctrl+C to stop the server\n")

    app.run(
        debug=True,
        host="127.0.0.1",
        port=8050,
    )
