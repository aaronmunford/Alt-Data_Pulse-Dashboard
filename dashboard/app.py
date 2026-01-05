import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Quantitative Signal Platform", layout="wide")

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600;700&family=IBM+Plex+Mono:wght@400;600&display=swap');

:root {
  --bg: #f3efe6;
  --card: #ffffff;
  --ink: #101213;
  --muted: #5a5f63;
  --accent: #0b6e4f;
  --accent-soft: rgba(11, 110, 79, 0.12);
  --warning: #c0542f;
  --line: rgba(16, 18, 19, 0.08);
  --shadow: 0 16px 30px rgba(16, 18, 19, 0.08);
}

html, body, [class*="css"]  {
  font-family: 'Space Grotesk', sans-serif;
  color: var(--ink);
}

.stApp {
  background:
    radial-gradient(circle at 10% 10%, #ffffff 0%, #f3efe6 35%, #f0e7d8 100%),
    linear-gradient(135deg, #f3efe6 0%, #f7f2e9 100%);
}

.hero-title {
  font-size: 3.2rem;
  font-weight: 700;
  letter-spacing: -0.03em;
  margin-bottom: 0.35rem;
}

.hero-subtitle {
  font-size: 1.1rem;
  color: var(--muted);
  max-width: 46rem;
}

.hero-tag {
  display: inline-flex;
  gap: 0.5rem;
  align-items: center;
  padding: 0.4rem 0.75rem;
  border-radius: 999px;
  background: var(--accent-soft);
  color: var(--accent);
  font-weight: 600;
  font-size: 0.85rem;
  text-transform: uppercase;
  letter-spacing: 0.08em;
}

.section-label {
  font-size: 1.2rem;
  font-weight: 600;
  margin: 1.6rem 0 0.6rem;
}

.alpha-card, .engine-card, .signal-card {
  background: var(--card);
  border-radius: 20px;
  border: 1px solid var(--line);
  box-shadow: var(--shadow);
  padding: 1.2rem 1.4rem;
}

.alpha-card {
  min-height: 140px;
}

.card-title {
  font-weight: 600;
  color: var(--muted);
  font-size: 0.95rem;
  text-transform: uppercase;
  letter-spacing: 0.06em;
}

.card-value {
  font-family: 'IBM Plex Mono', monospace;
  font-size: 2.1rem;
  font-weight: 600;
  margin: 0.35rem 0 0.25rem;
}

.card-note {
  color: var(--muted);
  font-size: 0.95rem;
}

.delta-pill {
  display: inline-flex;
  align-items: center;
  gap: 0.35rem;
  padding: 0.25rem 0.6rem;
  border-radius: 999px;
  font-weight: 600;
  font-size: 0.85rem;
}

.delta-up {
  background: rgba(11, 110, 79, 0.15);
  color: var(--accent);
}

.delta-down {
  background: rgba(192, 84, 47, 0.15);
  color: var(--warning);
}

.insight {
  background: #101213;
  color: #f9f6f0;
  padding: 1rem 1.4rem;
  border-radius: 16px;
  font-size: 1.05rem;
  margin-top: 1rem;
  box-shadow: var(--shadow);
}

.engine-card {
  min-height: 190px;
}

.engine-title {
  font-weight: 600;
  margin-bottom: 0.5rem;
  font-size: 1.05rem;
}

.engine-text {
  color: var(--muted);
  font-size: 0.98rem;
}
</style>
""",
    unsafe_allow_html=True,
)

live_revenue = 9.4
consensus_revenue = 9.1
delta_pct = (live_revenue - consensus_revenue) / consensus_revenue * 100
signal_strength = 0.92

months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
traffic_2023 = np.linspace(104, 100, 12)
traffic_2024 = np.linspace(103, 98, 12)
ticket_2023 = np.linspace(9.6, 10.0, 12)
ticket_2024 = np.linspace(9.7, 10.5, 12)

traffic_delta = (traffic_2024[-1] / traffic_2023[-1] - 1) * 100
ticket_delta = (ticket_2024[-1] / ticket_2023[-1] - 1) * 100

traffic_df = pd.DataFrame(
    {
        "Month": months,
        "Visits 2024": traffic_2024,
        "Visits 2023": traffic_2023,
    }
).melt("Month", var_name="Series", value_name="Value")

ticket_df = pd.DataFrame(
    {
        "Month": months,
        "Avg Ticket 2024": ticket_2024,
        "Avg Ticket 2023": ticket_2023,
    }
).melt("Month", var_name="Series", value_name="Value")

st.markdown(
    """
<div class="hero-tag">Quant Signal</div>
<div class="hero-title">Quantitative Signal Platform</div>
<div class="hero-subtitle">
Tracks real-time revenue performance for QSR equities against Wall Street consensus and surfaces the trade signal.
</div>
""",
    unsafe_allow_html=True,
)

hero_left, hero_right = st.columns([3, 1])
with hero_left:
    st.markdown("&nbsp;", unsafe_allow_html=True)
with hero_right:
    st.markdown(
        f"""
<div class="signal-card">
  <div class="card-title">Live Trade Signal</div>
  <div class="card-value">{delta_pct:+.1f}%</div>
  <div class="card-note">Revenue proxy vs consensus</div>
  <div class="delta-pill delta-up">Signal Strength: High ({signal_strength:.2f} corr)</div>
</div>
""",
        unsafe_allow_html=True,
    )

st.markdown('<div class="section-label">Alpha Metrics</div>', unsafe_allow_html=True)

alpha_cols = st.columns(4)
alpha_cols[0].markdown(
    """
<div class="alpha-card">
  <div class="card-title">Live Revenue Proxy</div>
  <div class="card-value">$9.4B</div>
  <div class="card-note">Predicted this quarter</div>
</div>
""",
    unsafe_allow_html=True,
)
alpha_cols[1].markdown(
    """
<div class="alpha-card">
  <div class="card-title">Wall St. Consensus</div>
  <div class="card-value">$9.1B</div>
  <div class="card-note">FactSet estimate</div>
</div>
""",
    unsafe_allow_html=True,
)
alpha_cols[2].markdown(
    f"""
<div class="alpha-card">
  <div class="card-title">The Delta</div>
  <div class="card-value">{delta_pct:+.1f}%</div>
  <div class="card-note">Trade signal</div>
  <div class="delta-pill delta-up">Beat</div>
</div>
""",
    unsafe_allow_html=True,
)
alpha_cols[3].markdown(
    f"""
<div class="alpha-card">
  <div class="card-title">Signal Strength</div>
  <div class="card-value">High</div>
  <div class="card-note">Based on {signal_strength:.2f} correlation</div>
</div>
""",
    unsafe_allow_html=True,
)

st.markdown('<div class="section-label">Why Revenue Is Moving</div>', unsafe_allow_html=True)

chart_cols = st.columns(2)

traffic_chart = (
    alt.Chart(traffic_df)
    .mark_line(point=True, strokeWidth=3)
    .encode(
        x=alt.X("Month:O", sort=months),
        y=alt.Y("Value:Q", title="Store Visits (Index)"),
        color=alt.Color(
            "Series:N",
            scale=alt.Scale(range=["#0b6e4f", "#c0542f"]),
        ),
        tooltip=["Month", "Series", "Value"],
    )
    .properties(height=280, title="Chart A: Foot Traffic (Advan)")
    .configure_view(strokeWidth=0)
    .configure_axis(
        grid=True,
        gridColor="rgba(16,18,19,0.08)",
        labelColor="#5a5f63",
        titleColor="#101213",
    )
    .configure_title(fontSize=16, font="Space Grotesk")
)

ticket_chart = (
    alt.Chart(ticket_df)
    .mark_line(point=True, strokeWidth=3)
    .encode(
        x=alt.X("Month:O", sort=months),
        y=alt.Y("Value:Q", title="Average Ticket ($)"),
        color=alt.Color(
            "Series:N",
            scale=alt.Scale(range=["#1f3a93", "#f09d51"]),
        ),
        tooltip=["Month", "Series", "Value"],
    )
    .properties(height=280, title="Chart B: Ticket Size (Consumer Edge)")
    .configure_view(strokeWidth=0)
    .configure_axis(
        grid=True,
        gridColor="rgba(16,18,19,0.08)",
        labelColor="#5a5f63",
        titleColor="#101213",
    )
    .configure_title(fontSize=16, font="Space Grotesk")
)

with chart_cols[0]:
    st.altair_chart(traffic_chart, use_container_width=True)
with chart_cols[1]:
    st.altair_chart(ticket_chart, use_container_width=True)

st.markdown(
    f"""
<div class="insight">
Traffic is down {traffic_delta:+.0f}%, but pricing is up {ticket_delta:+.0f}%, driving the {delta_pct:+.1f}% revenue beat.
</div>
""",
    unsafe_allow_html=True,
)

st.markdown('<div class="section-label">The Engine: What It Does Automatically</div>', unsafe_allow_html=True)

engine_cols = st.columns(3)
engine_cols[0].markdown(
    """
<div class="engine-card">
  <div class="engine-title">1. Ingests & Cleans</div>
  <div class="engine-text">
    Each week it pulls massive Dewey drops, explodes JSON arrays, and cleans the data end-to-end.
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
    It aligns odd fiscal quarters like "Oct 2 - Dec 31" so comparisons stay apples-to-apples.
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
    A live linear regression learns the 2023-2024 relationship between visits and revenue to predict next quarter.
  </div>
</div>
""",
    unsafe_allow_html=True,
)

st.caption("Sources: Advan foot traffic, Consumer Edge ticket size, FactSet consensus. Data shown is mocked for layout.")
