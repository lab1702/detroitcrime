"""Geography page — location-based crime analysis."""

import streamlit as st
import pandas as pd
import plotly.express as px

from constants import CHART_HEIGHT, CODE_LABEL_PREFIXES, TOP_N_ITEMS
from utils import code_labels
from theme import inject_css, apply_dark_plotly, RED_ACCENT, AMBER_ACCENT


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


def code_counts(df: pd.DataFrame, column: str) -> pd.Series:
    """Incidents per precinct, district or zip code, the most frequent first, by label.

    The codes were read as numbers when the data was loaded (see
    `clean_and_filter_data`), so they are only counted and labelled here. An
    incident without a valid code has none to be counted under.
    """
    counts = df[column].value_counts()
    counts.index = code_labels(counts.index, CODE_LABEL_PREFIXES[column])
    return counts


def render() -> None:
    """Draw the page.

    The data is kept in this function's own variables, not at the top level
    of the script: those would stay in the script's namespace, which the
    functions defined above keep in memory after the run has ended.
    """
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

    # ── Row 1: Neighborhoods + Precincts ────────────────────────────────────
    left, right = st.columns(2)

    with left:
        st.subheader(f"Top {TOP_N_ITEMS} Neighborhoods")
        counts = df_filtered["neighborhood"].value_counts().head(TOP_N_ITEMS)
        bar_chart(counts, "Neighborhood", RED_ACCENT)

    with right:
        # Every precinct, not a top 10: there are 11, so one would always be left out
        st.subheader("Police Precincts")
        bar_chart(code_counts(df_filtered, "police_precinct"), "Precinct", AMBER_ACCENT)

    # ── Row 2: Districts + Zip Codes ─────────────────────────────────────────
    left2, right2 = st.columns(2)

    with left2:
        # Every district as well: there are 7
        st.subheader("Council Districts")
        bar_chart(code_counts(df_filtered, "council_district"), "District", RED_ACCENT)

    with right2:
        st.subheader(f"Top {TOP_N_ITEMS} Zip Codes")
        bar_chart(code_counts(df_filtered, "zip_code").head(TOP_N_ITEMS), "Zip Code", AMBER_ACCENT)

    # ── Row 3: Top Locations (full width) ────────────────────────────────────
    st.subheader(f"Top {TOP_N_ITEMS} Locations")
    counts = df_filtered["nearest_intersection"].value_counts().head(TOP_N_ITEMS)
    bar_chart(counts, "Intersection", RED_ACCENT)


render()
