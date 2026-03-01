"""Overview page — KPI cards and summary charts."""

from datetime import datetime
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from constants import CHART_HEIGHT, TOP_N_ITEMS
from theme import inject_css, apply_dark_plotly, BLUE_ACCENT, GREEN_ACCENT, RED_ACCENT

inject_css()

st.header("Overview")

# Guard: data must be loaded by app.py
if "df_filtered" not in st.session_state:
    st.warning("Loading data — please wait…")
    st.stop()

df_filtered: pd.DataFrame = st.session_state["df_filtered"]
selected_category: str = st.session_state["selected_category"]

if df_filtered.empty:
    st.warning("No data available for selected filters.")
    st.stop()

# ── KPI Metrics ──────────────────────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)
min_date = df_filtered["date"].min()
max_date = df_filtered["date"].max()
total_days = (max_date - min_date).days + 1
avg_daily = len(df_filtered) / total_days if total_days > 0 else 0

with col1:
    st.metric("Total Incidents", f"{len(df_filtered):,}")
with col2:
    st.metric("Date Range", f"{total_days:,} days", f"{min_date} to {max_date}")
with col3:
    st.metric("Avg Daily", f"{avg_daily:.1f}")
with col4:
    if selected_category != "All":
        st.metric("Selected Category", selected_category)
    else:
        most_common = (
            df_filtered["offense_category"].mode()[0]
            if not df_filtered.empty
            else "N/A"
        )
        st.metric("Most Common", most_common)

st.divider()

# ── Summary Charts (side-by-side) ────────────────────────────────────────────
left, right = st.columns(2)

with left:
    if selected_category != "All":
        st.subheader(f"Top {TOP_N_ITEMS} Offense Descriptions")
        counts = df_filtered["offense_description"].value_counts().head(TOP_N_ITEMS)
    else:
        st.subheader(f"Top {TOP_N_ITEMS} Crime Categories")
        counts = df_filtered["offense_category"].value_counts().head(TOP_N_ITEMS)

    fig_cat = px.bar(
        x=counts.values[::-1],
        y=counts.index[::-1],
        orientation="h",
        labels={"x": "Incidents", "y": ""},
    )
    fig_cat.update_layout(height=CHART_HEIGHT)
    apply_dark_plotly(fig_cat, RED_ACCENT)
    st.plotly_chart(fig_cat, use_container_width=True)

with right:
    st.subheader("Case Status")
    status_counts = df_filtered["case_status"].value_counts()
    fig_status = px.bar(
        x=status_counts.values[::-1],
        y=status_counts.index[::-1],
        orientation="h",
        labels={"x": "Incidents", "y": ""},
    )
    fig_status.update_layout(height=CHART_HEIGHT)
    apply_dark_plotly(fig_status, GREEN_ACCENT)
    st.plotly_chart(fig_status, use_container_width=True)

# ── Daily Trend Sparkline ────────────────────────────────────────────────────
st.subheader("Daily Incidents")
daily = df_filtered.groupby("date").size().reset_index(name="count")
daily["date"] = pd.to_datetime(daily["date"])
date_range = pd.date_range(start=daily["date"].min(), end=daily["date"].max(), freq="D")
daily = pd.DataFrame({"date": date_range}).merge(daily, on="date", how="left").fillna(0)

fig_daily = px.line(daily, x="date", y="count", labels={"date": "", "count": "Incidents"})
fig_daily.update_layout(height=200)
apply_dark_plotly(fig_daily, BLUE_ACCENT)
st.plotly_chart(fig_daily, use_container_width=True)

# ── Footer ───────────────────────────────────────────────────────────────────
st.markdown(
    f'<p class="footer-text">Data source: Detroit Open Data Portal · '
    f'Last updated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>',
    unsafe_allow_html=True,
)
