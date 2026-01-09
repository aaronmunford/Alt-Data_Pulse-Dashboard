"""
Alt-Data Pulse Dashboard - Plotly Dash Version
Phase 2H: Production-Ready Bloomberg-Style Dashboard
"""

import dash
from dash import html, dcc, callback, Input, Output, State
import dash_bootstrap_components as dbc
import pandas as pd
import json
from datetime import datetime

# Import data utilities
from utils import (
    load_brand_signal,
    load_brand_trend_data,
    get_available_brands,
    format_currency,
    format_delta,
    get_delta_color,
    get_signal_color,
)

# Import chart builders
from charts import create_traffic_chart, create_ticket_chart, create_combined_chart

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
# Layout
# ============================================================================

app.layout = dbc.Container(
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
        # Combined Chart Section
        dbc.Row(
            [
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
                    width=12,
                )
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

# ============================================================================
# Callbacks
# ============================================================================

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
