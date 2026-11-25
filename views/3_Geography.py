"""Geography page — location-based crime analysis."""

import streamlit as st
import pandas as pd
import plotly.express as px

from constants import CHART_HEIGHT, TOP_N_ITEMS
from utils import format_categorical_series
from theme import inject_css, apply_dark_plotly, RED_ACCENT, AMBER_ACCENT

inject_css()

st.header("Geography")

# The data app.py lends each run (see app.main)
run_data = st.session_state.get("run_data", {})
if "df_filtered" not in run_data:
    st.warning("Loading data — please wait…")
    st.stop()

df_filtered: pd.DataFrame = run_data["df_filtered"]

if df_filtered.empty:
    st.warning("No data available for selected filters.")
    st.stop()


def bar_chart(counts: pd.Series, y_label: str, accent: str, height: int = CHART_HEIGHT) -> None:
    """Draw value counts as a styled horizontal bar chart, the largest on top.

    With nothing to count (no incident in the selection has the value) it
    says so instead: Plotly Express does not take empty data.
    """
    if counts.empty:
        st.info("Nothing recorded for this selection.")
        return
    fig = px.bar(
        x=counts.values[::-1],
        y=counts.index[::-1],
        orientation="h",
        labels={"x": "Incidents", "y": y_label},
    )
    fig.update_layout(height=height)
    fig.update_yaxes(type="category")
    apply_dark_plotly(fig, accent)
    st.plotly_chart(fig, width="stretch")


# ── Row 1: Neighborhoods + Precincts ────────────────────────────────────────
left, right = st.columns(2)

with left:
    st.subheader(f"Top {TOP_N_ITEMS} Neighborhoods")
    counts = df_filtered["neighborhood"].value_counts().head(TOP_N_ITEMS)
    bar_chart(counts, "Neighborhood", RED_ACCENT)

with right:
    # Every precinct, not a top 10: there are 11, so one would always be left out
    st.subheader("Police Precincts")
    precincts = format_categorical_series(df_filtered["police_precinct"], "Precinct")
    bar_chart(precincts.value_counts(), "Precinct", AMBER_ACCENT)


# ── Row 2: Districts + Zip Codes ─────────────────────────────────────────────
left2, right2 = st.columns(2)

with left2:
    # Every district as well: there are 7
    st.subheader("Council Districts")
    districts = format_categorical_series(df_filtered["council_district"], "District")
    bar_chart(districts.value_counts(), "District", RED_ACCENT)

with right2:
    st.subheader(f"Top {TOP_N_ITEMS} Zip Codes")
    zip_codes = format_categorical_series(df_filtered["zip_code"])
    counts = zip_codes.value_counts().head(TOP_N_ITEMS)
    bar_chart(counts, "Zip Code", AMBER_ACCENT)


# ── Row 3: Top Locations (full width) ────────────────────────────────────────
st.subheader(f"Top {TOP_N_ITEMS} Locations")
counts = df_filtered["nearest_intersection"].value_counts().head(TOP_N_ITEMS)
bar_chart(counts, "Intersection", RED_ACCENT)
