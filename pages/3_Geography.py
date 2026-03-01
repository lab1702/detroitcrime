"""Geography page — location-based crime analysis."""

import streamlit as st
import pandas as pd
import plotly.express as px

from constants import CHART_HEIGHT, TOP_N_ITEMS
from utils import format_categorical_column
from theme import inject_css, apply_dark_plotly, RED_ACCENT, AMBER_ACCENT

inject_css()

st.header("Geography")

if "df_filtered" not in st.session_state:
    st.warning("Loading data — please wait…")
    st.stop()

df_filtered: pd.DataFrame = st.session_state["df_filtered"]

if df_filtered.empty:
    st.warning("No data available for selected filters.")
    st.stop()


def bar_chart(counts, y_label: str, accent: str, height: int = CHART_HEIGHT):
    """Create a styled horizontal bar chart from value counts."""
    fig = px.bar(
        x=counts.values[::-1],
        y=counts.index[::-1],
        orientation="h",
        labels={"x": "Incidents", "y": y_label},
    )
    fig.update_layout(height=height)
    fig.update_yaxes(type="category")
    apply_dark_plotly(fig, accent)
    return fig


# ── Row 1: Neighborhoods + Precincts ────────────────────────────────────────
left, right = st.columns(2)

with left:
    st.subheader(f"Top {TOP_N_ITEMS} Neighborhoods")
    counts = df_filtered["neighborhood"].value_counts().head(TOP_N_ITEMS)
    st.plotly_chart(bar_chart(counts, "Neighborhood", RED_ACCENT), use_container_width=True)

with right:
    st.subheader(f"Top {TOP_N_ITEMS} Police Precincts")
    df_fmt = format_categorical_column(df_filtered, "police_precinct", "Precinct")
    counts = df_fmt["police_precinct"].value_counts().head(TOP_N_ITEMS)
    st.plotly_chart(bar_chart(counts, "Precinct", AMBER_ACCENT), use_container_width=True)


# ── Row 2: Districts + Zip Codes ─────────────────────────────────────────────
left2, right2 = st.columns(2)

with left2:
    st.subheader(f"Top {TOP_N_ITEMS} Council Districts")
    df_fmt = format_categorical_column(df_filtered, "council_district", "District")
    counts = df_fmt["council_district"].value_counts().head(TOP_N_ITEMS)
    st.plotly_chart(bar_chart(counts, "District", RED_ACCENT), use_container_width=True)

with right2:
    st.subheader(f"Top {TOP_N_ITEMS} Zip Codes")
    df_zip = df_filtered.copy()
    df_zip["zip_code"] = (
        pd.to_numeric(df_zip["zip_code"], errors="coerce").fillna(0).astype(int).astype(str)
    )
    df_zip = df_zip[df_zip["zip_code"] != "0"]
    counts = df_zip["zip_code"].value_counts().head(TOP_N_ITEMS)
    st.plotly_chart(bar_chart(counts, "Zip Code", AMBER_ACCENT), use_container_width=True)


# ── Row 3: Top Locations (full width) ────────────────────────────────────────
st.subheader(f"Top {TOP_N_ITEMS} Locations")
counts = df_filtered["nearest_intersection"].value_counts().head(TOP_N_ITEMS)
st.plotly_chart(bar_chart(counts, "Intersection", RED_ACCENT), use_container_width=True)
