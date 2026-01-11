"""
Quantitative Signal Platform - Dashboard
=========================================
Real-time revenue tracking for QSR stocks vs Wall Street consensus.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from datetime import datetime, timedelta

# Import the prediction model
try:
    from ingest.predictor import RevenuePredictor
    USE_REAL_DATA = True
except ImportError:
    USE_REAL_DATA = False

st.set_page_config(page_title="Quantitative Signal Platform", layout="wide")

# =============================================================================
# DARK THEME STYLING
# =============================================================================
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600;700&family=IBM+Plex+Mono:wght@400;600&display=swap');

:root {
  --bg: #0d1117;
  --bg-secondary: #161b22;
  --card: #1c2128;
  --card-hover: #21262d;
  --border: #30363d;
  --text-primary: #e6edf3;
  --text-secondary: #8b949e;
  --text-muted: #6e7681;
  --accent: #3fb950;
  --accent-soft: rgba(63, 185, 80, 0.15);
  --warning: #f85149;
  --warning-soft: rgba(248, 81, 73, 0.15);
  --blue: #58a6ff;
  --orange: #d29922;
  --shadow: 0 8px 24px rgba(0, 0, 0, 0.4);
}

html, body, [class*="css"] {
  font-family: 'Space Grotesk', sans-serif;
  color: var(--text-primary);
}

.stApp {
  background: var(--bg);
}

/* Hide default Streamlit elements */
#MainMenu, footer, header {visibility: hidden;}

/* Main container */
.main .block-container {
  padding-top: 2rem;
  max-width: 1400px;
}

/* Hero Section */
.hero-tag {
  display: inline-flex;
  padding: 0.4rem 0.75rem;
  border-radius: 999px;
  background: var(--accent-soft);
  color: var(--accent);
  font-weight: 600;
  font-size: 0.8rem;
  text-transform: uppercase;
  letter-spacing: 0.1em;
  margin-bottom: 0.75rem;
}

.hero-title {
  font-size: 2.8rem;
  font-weight: 700;
  letter-spacing: -0.02em;
  margin-bottom: 0.5rem;
  color: var(--text-primary);
}

.hero-subtitle {
  font-size: 1.05rem;
  color: var(--text-secondary);
  max-width: 600px;
  line-height: 1.5;
}

/* Section Labels */
.section-label {
  font-size: 1rem;
  font-weight: 600;
  margin: 2rem 0 1rem;
  color: var(--text-secondary);
  text-transform: uppercase;
  letter-spacing: 0.08em;
  display: flex;
  align-items: center;
  gap: 1rem;
}

.section-label::after {
  content: "";
  flex: 1;
  height: 1px;
  background: var(--border);
}

/* Cards */
.alpha-card, .engine-card, .signal-card {
  background: var(--card);
  border-radius: 12px;
  border: 1px solid var(--border);
  padding: 1.25rem;
  transition: background 0.2s;
}

.alpha-card:hover, .engine-card:hover {
  background: var(--card-hover);
}

.signal-card {
  background: linear-gradient(135deg, var(--card) 0%, #1a2332 100%);
  border-color: var(--accent);
}

.card-title {
  font-weight: 600;
  color: var(--text-muted);
  font-size: 0.75rem;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  margin-bottom: 0.5rem;
}

.card-value {
  font-family: 'IBM Plex Mono', monospace;
  font-size: 1.9rem;
  font-weight: 600;
  color: var(--text-primary);
  margin-bottom: 0.25rem;
}

.card-note {
  color: var(--text-secondary);
  font-size: 0.85rem;
}

.card-subnote {
  color: var(--text-muted);
  font-size: 0.75rem;
  margin-top: 0.35rem;
}

.lseg-badge {
  display: inline-flex;
  align-items: center;
  gap: 0.4rem;
  padding: 0.25rem 0.6rem;
  border-radius: 999px;
  background: rgba(88, 166, 255, 0.15);
  color: var(--blue);
  font-size: 0.7rem;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.08em;
}

.consensus-meta {
  margin-top: 0.75rem;
  color: var(--text-secondary);
  font-size: 0.85rem;
}

/* Delta Pills */
.delta-pill {
  display: inline-flex;
  align-items: center;
  padding: 0.25rem 0.6rem;
  border-radius: 999px;
  font-weight: 600;
  font-size: 0.75rem;
  margin-top: 0.5rem;
}

.delta-up {
  background: var(--accent-soft);
  color: var(--accent);
}

.delta-down {
  background: var(--warning-soft);
  color: var(--warning);
}

/* Insight Box */
.insight {
  background: linear-gradient(135deg, #1a2332 0%, #0d1926 100%);
  border: 1px solid var(--border);
  color: var(--text-primary);
  padding: 1rem 1.25rem;
  border-radius: 12px;
  font-size: 0.95rem;
  margin-top: 1rem;
}

.insight strong {
  color: var(--orange);
}

/* Engine Cards */
.engine-card {
  min-height: 160px;
}

.engine-title {
  font-weight: 600;
  margin-bottom: 0.5rem;
  font-size: 1rem;
  color: var(--text-primary);
}

.engine-text {
  color: var(--text-secondary);
  font-size: 0.9rem;
  line-height: 1.5;
}

/* Mock Data Banner */
.mock-banner {
  background: var(--warning-soft);
  border: 1px solid var(--warning);
  color: var(--warning);
  padding: 0.5rem 1rem;
  border-radius: 8px;
  font-size: 0.85rem;
  font-weight: 600;
  margin-bottom: 1rem;
  display: inline-flex;
  align-items: center;
  gap: 0.5rem;
}

/* Selectbox Styling */
div[data-testid="stSelectbox"] label {
  color: var(--text-secondary) !important;
  text-transform: uppercase;
  letter-spacing: 0.1em;
  font-size: 0.7rem;
  font-weight: 600;
}

div[data-baseweb="select"] > div {
  background: var(--card) !important;
  border: 1px solid var(--border) !important;
  border-radius: 8px !important;
}

div[data-baseweb="select"] > div:hover {
  border-color: var(--text-muted) !important;
}

div[data-baseweb="select"] span {
  color: var(--text-primary) !important;
  font-weight: 600;
}

div[data-baseweb="popover"] > div {
  background: var(--card) !important;
  border: 1px solid var(--border) !important;
}

div[role="listbox"] {
  background: var(--card) !important;
}

div[role="option"] {
  color: var(--text-primary) !important;
}

div[role="option"]:hover {
  background: var(--card-hover) !important;
}

/* Chart Container */
div[data-testid="stVegaLiteChart"] {
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 1rem;
  overflow: hidden;
}

/* Caption */
.stCaption {
  color: var(--text-muted) !important;
}

/* Footer */
.footer-info {
  color: var(--text-muted);
  font-size: 0.8rem;
  padding: 1rem 0;
  border-top: 1px solid var(--border);
  margin-top: 2rem;
}
</style>
""",
    unsafe_allow_html=True,
)

# =============================================================================
# DATA LOADING
# =============================================================================
@st.cache_data(ttl=300)
def load_data(brand: str):
    """Load and process data for the selected brand."""
    if USE_REAL_DATA:
        try:
          predictor = RevenuePredictor(
            spend_path='data/clean_spend_daily.csv',
            traffic_path='data/clean_traffic_daily.csv',
            retail_sales_path='data/market_data_retail_sales.csv',
          )
          signal = predictor.predict_revenue(brand)
          trend_df = predictor.get_trend_data(brand)
          return signal, trend_df, True
        except Exception as e:
          st.error(f"Error loading real data: {e}")
          pass  # Fall through to mock data

    # Mock data fallback
    signal = {
        'brand': brand,
        'quarter': '2025_Q1',
        'predicted_revenue': 9.4e9,
        'wall_street_consensus': 9.1e9,
        'delta_pct': 3.3,
        'delta_direction': 'BEAT',
        'signal_strength': 'High',
        'correlation': 0.92,
        'consensus_source': 'mock',
        'consensus_revenue_high': 9.4e9,
        'consensus_revenue_low': 8.9e9,
        'consensus_num_analysts': 26,
        'consensus_earnings_date': (datetime.now().date()).isoformat(),
        'consensus_fetched_at': datetime.now().isoformat(),
        'consensus_is_stale': False,
    }
    return signal, pd.DataFrame(), False


def get_chart_data(trend_df: pd.DataFrame, date_range_days: int = 365):
    """Prepare data for charts using daily data with 7-day rolling averages.

    Args:
        trend_df: DataFrame with daily data from predictor.get_trend_data()
        date_range_days: Number of days to show (default: 365 for last year)
    """
    if trend_df.empty:
        # Generate mock data for fallback (daily data for last 90 days)
        dates = pd.date_range(end=datetime.now(), periods=90, freq='D')

        traffic_df = pd.DataFrame({
            "date": dates,
            "Value": np.linspace(100, 98, 90) + np.random.randn(90) * 2,
        })

        ticket_df = pd.DataFrame({
            "date": dates,
            "Value": np.linspace(9.50, 10.50, 90) + np.random.randn(90) * 0.15,
        })

        return traffic_df, ticket_df, -2.0, 5.0

    # Use real data
    df = trend_df.copy()

    # Filter to specified date range (use data's max date as reference, not system date)
    max_date = df['date'].max()
    cutoff_date = max_date - timedelta(days=date_range_days)
    df = df[df['date'] >= cutoff_date]

    # Traffic: Use 7d rolling avg of visits (visits_7d_avg)
    if 'visits_7d_avg' in df.columns:
        traffic_df = df[['date', 'visits_7d_avg']].copy()
        traffic_df = traffic_df.dropna(subset=['visits_7d_avg'])
        # Normalize to index (mean = 100)
        if len(traffic_df) > 0 and traffic_df['visits_7d_avg'].mean() > 0:
            traffic_df['Value'] = (traffic_df['visits_7d_avg'] /
                                    traffic_df['visits_7d_avg'].mean() * 100)
        else:
            traffic_df['Value'] = 100
    else:
        # Fallback to transactions if visits not available
        traffic_df = df[['date', 'transactions_7d_avg']].copy() if 'transactions_7d_avg' in df.columns else df[['date', 'transactions']].copy()
        traffic_df = traffic_df.dropna()
        col_name = 'transactions_7d_avg' if 'transactions_7d_avg' in traffic_df.columns else 'transactions'
        if len(traffic_df) > 0 and traffic_df[col_name].mean() > 0:
            traffic_df['Value'] = (traffic_df[col_name] / traffic_df[col_name].mean() * 100)
        else:
            traffic_df['Value'] = 100

    # Ticket Size: Use 7d smoothed avg_ticket_size
    ticket_df = df[['date', 'avg_ticket_size']].copy()
    ticket_df = ticket_df.dropna(subset=['avg_ticket_size'])
    # Apply 7-day rolling average for smoothness
    ticket_df['Value'] = ticket_df['avg_ticket_size'].rolling(7, min_periods=1).mean()

    # Calculate YoY deltas for KPI cards (compare latest week to same week last year)
    # Use original unfiltered data for YoY comparison
    full_df = trend_df.copy()

    # Get latest 7 days from the filtered data
    latest_week = df.tail(7)

    # Get same week from last year (365 days ago) - use full_df to ensure we have historical data
    year_ago_start = latest_week['date'].min() - timedelta(days=365)
    year_ago_end = latest_week['date'].max() - timedelta(days=365)
    year_ago_week = full_df[full_df['date'].between(year_ago_start, year_ago_end)]

    if not latest_week.empty and not year_ago_week.empty:
        # For traffic delta, use visits if available, otherwise transactions
        if 'total_visits' in latest_week.columns:
            traffic_latest = latest_week['total_visits'].mean()
            traffic_year_ago = year_ago_week['total_visits'].mean()
        else:
            traffic_latest = latest_week['transactions'].mean()
            traffic_year_ago = year_ago_week['transactions'].mean()

        if traffic_year_ago > 0:
            traffic_delta = ((traffic_latest / traffic_year_ago) - 1) * 100
        else:
            traffic_delta = 0.0

        # For ticket size delta
        ticket_latest = latest_week['avg_ticket_size'].mean()
        ticket_year_ago = year_ago_week['avg_ticket_size'].mean()

        if ticket_year_ago > 0:
            ticket_delta = ((ticket_latest / ticket_year_ago) - 1) * 100
        else:
            ticket_delta = 0.0
    else:
        traffic_delta, ticket_delta = -2.0, 5.0

    return traffic_df, ticket_df, traffic_delta, ticket_delta


# =============================================================================
# HEADER
# =============================================================================
st.markdown(
    """
<div class="hero-tag">Quant Signal</div>
<div class="hero-title">Quantitative Signal Platform</div>
<div class="hero-subtitle">
Real-time revenue tracking for QSR equities against Wall Street consensus.
</div>
""",
    unsafe_allow_html=True,
)

# Brand selector
available_brands = ["STARBUCKS", "MCDONALD'S", "CHIPOTLE", "BURGER KING", "TACO BELL"]
selected_brand = st.selectbox("Select Brand", available_brands, index=0)

# Load data
signal, trend_df, is_live_data = load_data(selected_brand)

# Show mock data warning
if not is_live_data:
    st.markdown(
        '<div class="mock-banner">Demo Mode - Showing simulated data</div>',
        unsafe_allow_html=True
    )

# Extract values
if 'error' in signal:
    st.error(signal['error'])
    st.stop()

live_revenue = signal.get('predicted_revenue', 9.4e9) / 1e9
consensus_revenue = signal.get('wall_street_consensus', 9.1e9) / 1e9
consensus_high = signal.get('consensus_revenue_high')
consensus_low = signal.get('consensus_revenue_low')
consensus_num_analysts = signal.get('consensus_num_analysts')
consensus_source = signal.get('consensus_source', 'model_prediction')
consensus_fetched_at = signal.get('consensus_fetched_at')
consensus_is_stale = signal.get('consensus_is_stale', False)
consensus_earnings_date = signal.get('consensus_earnings_date')
delta_pct = signal.get('delta_pct', 3.3)
correlation = signal.get('correlation', 0.92)
signal_strength = signal.get('signal_strength', 'High')
quarter = signal.get('quarter', '2025_Q1')

# Map consensus source codes into something human-friendly for the UI.
source_labels = {
    "lseg_live": "LSEG SmartEstimate (live)",
    "lseg_cache": "LSEG SmartEstimate (cached)",
    "lseg_cache_stale": "LSEG SmartEstimate (stale)",
    "historical_growth": "Historical growth estimate",
    "historical_actual": "Historical actual",
    "model_prediction": "Model-derived consensus",
    "mock": "Mock consensus",
}
consensus_source_label = source_labels.get(consensus_source, consensus_source)

# Format earnings countdown in days for quick analyst context.
earnings_label = None
earnings_dt = pd.to_datetime(consensus_earnings_date, errors="coerce")
if pd.notna(earnings_dt):
    days_to_earnings = (earnings_dt.date() - datetime.now().date()).days
    date_label = earnings_dt.strftime("%b %d, %Y")
    if days_to_earnings >= 0:
        earnings_label = f"Earnings in {days_to_earnings} days ({date_label})"
    else:
        earnings_label = f"Earnings {abs(days_to_earnings)} days ago ({date_label})"

# =============================================================================
# LIVE SIGNAL CARD
# =============================================================================
_, signal_col = st.columns([3, 1])
with signal_col:
    delta_class = "delta-up" if delta_pct >= 0 else "delta-down"
    earnings_line = f'<div class="card-subnote">{earnings_label}</div>' if earnings_label else ""
    st.markdown(
        f"""
<div class="signal-card">
  <div class="card-title">Live Trade Signal</div>
  <div class="card-value">{delta_pct:+.1f}%</div>
  <div class="card-note">Revenue proxy vs consensus</div>
  {earnings_line}
  <div class="delta-pill {delta_class}">Signal: {signal_strength} ({correlation:.2f})</div>
</div>
""",
        unsafe_allow_html=True,
    )

# =============================================================================
# ALPHA METRICS
# =============================================================================
st.markdown('<div class="section-label">Alpha Metrics</div>', unsafe_allow_html=True)

range_note = "Range unavailable"
if consensus_low is not None and consensus_high is not None:
    range_note = f"Range ${consensus_low / 1e9:.2f}B - ${consensus_high / 1e9:.2f}B"

analyst_note = "Analyst count unavailable"
if consensus_num_analysts:
    analyst_note = f"{consensus_num_analysts} analysts"

stale_note = "Consensus stale - using fallback" if consensus_is_stale else ""

cols = st.columns(4)
cols[0].markdown(
    f"""
<div class="alpha-card">
  <div class="card-title">Live Revenue Proxy</div>
  <div class="card-value">${live_revenue:.2f}B</div>
  <div class="card-note">Predicted {quarter}</div>
</div>
""",
    unsafe_allow_html=True,
)
cols[1].markdown(
    f"""
<div class="alpha-card">
  <div class="card-title">Wall St. Consensus</div>
  <div class="card-value">${consensus_revenue:.2f}B</div>
  <div class="card-note">{consensus_source_label}</div>
  <div class="card-subnote">{range_note}</div>
  <div class="card-subnote">{analyst_note}</div>
  <div class="card-subnote">{stale_note}</div>
</div>
""",
    unsafe_allow_html=True,
)

delta_class = "delta-up" if delta_pct >= 0 else "delta-down"
delta_label = "Beat" if delta_pct >= 0 else "Miss"
cols[2].markdown(
    f"""
<div class="alpha-card">
  <div class="card-title">The Delta</div>
  <div class="card-value">{delta_pct:+.1f}%</div>
  <div class="card-note">Trade signal</div>
  <div class="delta-pill {delta_class}">{delta_label}</div>
</div>
""",
    unsafe_allow_html=True,
)
cols[3].markdown(
    f"""
<div class="alpha-card">
  <div class="card-title">Signal Strength</div>
  <div class="card-value">{signal_strength}</div>
  <div class="card-note">{correlation:.2f} correlation</div>
</div>
""",
    unsafe_allow_html=True,
)

# =============================================================================
# CHARTS
# =============================================================================
st.markdown('<div class="section-label">Why Revenue Is Moving</div>', unsafe_allow_html=True)

# Add date range selector
date_range_options = {
    "Last 90 Days": 90,
    "Last 6 Months": 180,
    "Last Year": 365,
    "All Available": 9999
}
selected_range = st.selectbox(
    "Time Range",
    options=list(date_range_options.keys()),
    index=2  # Default to "Last Year"
)
date_range_days = date_range_options[selected_range]

traffic_df, ticket_df, traffic_delta, ticket_delta = get_chart_data(trend_df, date_range_days)

# Dark theme colors for charts
chart_bg = "#1c2128"
grid_color = "#30363d"
text_color = "#8b949e"
title_color = "#e6edf3"

def build_chart(df, title, y_title, color_single, fmt):
    """Build an Altair time series chart with temporal encoding.

    Args:
        df: DataFrame with 'date' and 'Value' columns
        title: Chart title
        y_title: Y-axis title
        color_single: Single color for the line (hex string)
        fmt: Format string for tooltip values
    """
    return (
        alt.Chart(df)
        .mark_line(strokeWidth=2.5)  # Removed point=True for smoother lines
        .encode(
            x=alt.X("date:T",
                    title="Date",
                    axis=alt.Axis(
                        format="%b %Y",  # e.g., "Jan 2024"
                        labelAngle=-45,
                        labelColor=text_color,
                        titleColor=title_color,
                        gridColor=grid_color,
                        domainColor=grid_color,
                    )),
            y=alt.Y("Value:Q",
                    title=y_title,
                    scale=alt.Scale(zero=False),
                    axis=alt.Axis(
                        labelColor=text_color,
                        titleColor=title_color,
                        gridColor=grid_color,
                        domainColor=grid_color,
                    )),
            color=alt.value(color_single),  # Single color
            tooltip=[
                alt.Tooltip("date:T", title="Date", format="%b %d, %Y"),
                alt.Tooltip("Value:Q", title=y_title, format=fmt)
            ],
        )
        .properties(height=280, title=alt.Title(title, color=title_color))
        .configure_view(fill=chart_bg, strokeOpacity=0)
        .configure_axis(labelFont="IBM Plex Mono", titleFont="Space Grotesk",
                        labelFontSize=11, titleFontSize=12)
        .configure_title(font="Space Grotesk", fontSize=14, anchor="start")
    )

chart_cols = st.columns(2)

with chart_cols[0]:
    traffic_chart = build_chart(
        traffic_df, "Foot Traffic (Proxy)", "Store Visits (Index)",
        "#3fb950", ".1f"  # Single color: green
    )
    st.altair_chart(traffic_chart, use_container_width=True)

with chart_cols[1]:
    ticket_chart = build_chart(
        ticket_df, "Ticket Size (Consumer Edge)", "Avg Ticket ($)",
        "#58a6ff", "$,.2f"  # Single color: blue
    )
    st.altair_chart(ticket_chart, use_container_width=True)

# =============================================================================
# INSIGHT
# =============================================================================
traffic_dir = "down" if traffic_delta < 0 else "up"
ticket_dir = "up" if ticket_delta > 0 else "down"
revenue_dir = "beat" if delta_pct > 0 else "miss"

st.markdown(
    f"""
<div class="insight">
<strong>{selected_brand}:</strong> Traffic is {traffic_dir} {abs(traffic_delta):.0f}%, but ticket size is {ticket_dir} {abs(ticket_delta):.0f}%, driving the {delta_pct:+.1f}% revenue {revenue_dir}.
</div>
""",
    unsafe_allow_html=True,
)

# =============================================================================
# ENGINE SECTION
# =============================================================================
st.markdown('<div class="section-label">The Engine</div>', unsafe_allow_html=True)

engine_cols = st.columns(3)
engine_cols[0].markdown(
    """
<div class="engine-card">
  <div class="engine-title">1. Ingests & Cleans</div>
  <div class="engine-text">
    Pulls Dewey data weekly, explodes JSON arrays, and cleans everything automatically.
  </div>
</div>
""",
    unsafe_allow_html=True,
)
engine_cols[1].markdown(
    """
<div class="engine-card">
  <div class="engine-title">2. Harmonizes</div>
  <div class="engine-text">
    Aligns fiscal quarters (e.g., Starbucks Oct-Sep) so comparisons are apples-to-apples.
  </div>
</div>
""",
    unsafe_allow_html=True,
)
engine_cols[2].markdown(
    """
<div class="engine-card">
  <div class="engine-title">3. Predicts</div>
  <div class="engine-text">
    Linear regression learns spend-to-revenue relationship and predicts next quarter.
  </div>
</div>
""",
    unsafe_allow_html=True,
)

# Footer
mode = "Live" if is_live_data else "Demo"
consensus_asof = ""
if consensus_fetched_at:
    parsed_asof = pd.to_datetime(consensus_fetched_at, errors="coerce")
    if pd.notna(parsed_asof):
        consensus_asof = parsed_asof.strftime("%Y-%m-%d %H:%M")

is_lseg_source = consensus_source.startswith("lseg")
lseg_badge = '<span class="lseg-badge">Powered by LSEG</span>' if is_lseg_source else ""

st.markdown(
    f"""
<div class="footer-info">
Sources: Consumer Edge spend data | Consensus: {consensus_source_label} | Mode: {mode} | Updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}
<div class="consensus-meta">Consensus as of: {consensus_asof or "n/a"} {lseg_badge}</div>
</div>
""",
    unsafe_allow_html=True,
)
