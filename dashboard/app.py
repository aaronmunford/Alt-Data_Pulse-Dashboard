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
from datetime import datetime

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
            spend_path='../data/clean_spend_daily.csv',
            traffic_path='../data/clean_traffic_daily.csv'
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
    }
    return signal, pd.DataFrame(), False


def get_chart_data(trend_df: pd.DataFrame):
    """Prepare data for charts."""
    if trend_df.empty:
        # Generate mock data
        months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                  "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

        traffic_2023 = [100, 102, 104, 103, 101, 99, 98, 100, 102, 104, 103, 100]
        traffic_2024 = [98, 99, 101, 100, 98, 96, 95, 97, 99, 101, 100, 98]
        ticket_2023 = [9.50, 9.60, 9.70, 9.75, 9.80, 9.85, 9.90, 9.95, 10.00, 10.05, 10.10, 10.00]
        ticket_2024 = [9.70, 9.85, 10.00, 10.10, 10.20, 10.30, 10.40, 10.45, 10.50, 10.55, 10.60, 10.50]

        traffic_df = pd.DataFrame({
            "Month": months * 2,
            "Year": ["2023"] * 12 + ["2024"] * 12,
            "Value": traffic_2023 + traffic_2024,
        })

        ticket_df = pd.DataFrame({
            "Month": months * 2,
            "Year": ["2023"] * 12 + ["2024"] * 12,
            "Value": ticket_2023 + ticket_2024,
        })

        return traffic_df, ticket_df, -2.0, 5.0

    # Use real data
    trend_df = trend_df.copy()
    trend_df['month'] = trend_df['date'].dt.strftime('%b')
    trend_df['year'] = trend_df['date'].dt.year.astype(str)

    current_year = datetime.now().year
    recent = trend_df[trend_df['date'].dt.year >= current_year - 1]

    monthly = recent.groupby(['month', 'year']).agg({
        'transactions': 'sum',
        'avg_ticket_size': 'mean'
    }).reset_index()

    if len(monthly) > 0:
        latest = monthly[monthly['year'] == str(current_year)].tail(1)
        prev = monthly[monthly['year'] == str(current_year - 1)].tail(1)

        if not latest.empty and not prev.empty:
            traffic_delta = ((latest['transactions'].values[0] / prev['transactions'].values[0]) - 1) * 100
            ticket_delta = ((latest['avg_ticket_size'].values[0] / prev['avg_ticket_size'].values[0]) - 1) * 100
        else:
            traffic_delta, ticket_delta = -2.0, 5.0
    else:
        traffic_delta, ticket_delta = -2.0, 5.0

    monthly['transactions_idx'] = (monthly['transactions'] / monthly['transactions'].mean() * 100)

    traffic_df = monthly[['month', 'year', 'transactions_idx']].rename(
        columns={'month': 'Month', 'year': 'Year', 'transactions_idx': 'Value'})
    ticket_df = monthly[['month', 'year', 'avg_ticket_size']].rename(
        columns={'month': 'Month', 'year': 'Year', 'avg_ticket_size': 'Value'})

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
delta_pct = signal.get('delta_pct', 3.3)
correlation = signal.get('correlation', 0.92)
signal_strength = signal.get('signal_strength', 'High')
quarter = signal.get('quarter', '2025_Q1')

# =============================================================================
# LIVE SIGNAL CARD
# =============================================================================
_, signal_col = st.columns([3, 1])
with signal_col:
    delta_class = "delta-up" if delta_pct >= 0 else "delta-down"
    st.markdown(
        f"""
<div class="signal-card">
  <div class="card-title">Live Trade Signal</div>
  <div class="card-value">{delta_pct:+.1f}%</div>
  <div class="card-note">Revenue proxy vs consensus</div>
  <div class="delta-pill {delta_class}">Signal: {signal_strength} ({correlation:.2f})</div>
</div>
""",
        unsafe_allow_html=True,
    )

# =============================================================================
# ALPHA METRICS
# =============================================================================
st.markdown('<div class="section-label">Alpha Metrics</div>', unsafe_allow_html=True)

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
  <div class="card-note">FactSet estimate</div>
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

traffic_df, ticket_df, traffic_delta, ticket_delta = get_chart_data(trend_df)

months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

# Dark theme colors for charts
chart_bg = "#1c2128"
grid_color = "#30363d"
text_color = "#8b949e"
title_color = "#e6edf3"

def build_chart(df, title, y_title, colors, fmt):
    return (
        alt.Chart(df)
        .mark_line(point=True, strokeWidth=2.5)
        .encode(
            x=alt.X("Month:O", sort=months, axis=alt.Axis(
                labelColor=text_color,
                titleColor=title_color,
                gridColor=grid_color,
                domainColor=grid_color,
            )),
            y=alt.Y("Value:Q", title=y_title, scale=alt.Scale(zero=False), axis=alt.Axis(
                labelColor=text_color,
                titleColor=title_color,
                gridColor=grid_color,
                domainColor=grid_color,
            )),
            color=alt.Color("Year:N", scale=alt.Scale(range=colors),
                          legend=alt.Legend(orient="top", title=None, labelColor=text_color)),
            tooltip=["Month", "Year", alt.Tooltip("Value:Q", format=fmt)],
        )
        .properties(height=280, title=alt.Title(title, color=title_color))
        .configure_view(fill=chart_bg, strokeOpacity=0)
        .configure_axis(labelFont="IBM Plex Mono", titleFont="Space Grotesk", labelFontSize=11, titleFontSize=12)
        .configure_title(font="Space Grotesk", fontSize=14, anchor="start")
        .configure_legend(labelFont="IBM Plex Mono", labelFontSize=11)
        .configure_point(size=60)
    )

chart_cols = st.columns(2)

with chart_cols[0]:
    traffic_chart = build_chart(
        traffic_df, "Foot Traffic (Proxy)", "Store Visits (Index)",
        ["#3fb950", "#f85149"], ".1f"
    )
    st.altair_chart(traffic_chart, use_container_width=True)

with chart_cols[1]:
    ticket_chart = build_chart(
        ticket_df, "Ticket Size (Consumer Edge)", "Avg Ticket ($)",
        ["#58a6ff", "#d29922"], "$,.2f"
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
st.markdown(
    f"""
<div class="footer-info">
Sources: Consumer Edge spend data, FactSet consensus estimates | Mode: {mode} | Updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}
</div>
""",
    unsafe_allow_html=True,
)
