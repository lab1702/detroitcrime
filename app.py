"""Detroit Crime Incidents Dashboard

Entry point for the multi-page Streamlit application.
Handles data loading, caching, sidebar filters, session state,
and displays the Overview page (KPIs, summary charts, daily sparkline).
"""

from datetime import date

import streamlit as st
import pandas as pd
import plotly.express as px
import requests

from constants import (
    DATA_URL, REQUIRED_COLUMNS,
    MIN_DATE, DETROIT_LAT_MIN, DETROIT_LAT_MAX, DETROIT_LON_MIN, DETROIT_LON_MAX,
    CHART_HEIGHT, TOP_N_ITEMS,
)
from utils import (
    DataNotReadyError, validate_data, safe_load_data_from_url, clean_and_filter_data,
    compute_yoy_deltas, daily_incident_counts, dataset_cache, selection_period,
)
from theme import inject_css, apply_dark_plotly, BLUE_ACCENT, GREEN_ACCENT, RED_ACCENT

# Configure page
st.set_page_config(
    page_title="Detroit Crime Incidents",
    page_icon="\U0001f694",
    layout="wide",
)

inject_css()


def fetch_data() -> pd.DataFrame:
    """Download, validate and clean the Detroit crime data. Raises on any failure.

    Called through `dataset_cache()`, which keeps the result for every session
    and rerun to share, and keeps the previous result when a refresh fails.
    """
    df = safe_load_data_from_url(DATA_URL, usecols=REQUIRED_COLUMNS)
    validate_data(df)
    lat_bounds = (DETROIT_LAT_MIN, DETROIT_LAT_MAX)
    lon_bounds = (DETROIT_LON_MIN, DETROIT_LON_MAX)
    df = clean_and_filter_data(df, MIN_DATE, lat_bounds, lon_bounds)
    if df.empty:
        # A feed whose dates or coordinates changed can pass validation and
        # still lose every row here. Fail rather than keep an empty dataset.
        raise ValueError(
            "no incidents were left after cleaning; "
            "the dates or coordinates in the feed may have changed"
        )
    return df


def load_error_message(error: Exception) -> str:
    """Describe a failed data load for the user."""
    if isinstance(error, DataNotReadyError):
        return (
            f"The data portal did not send the dataset ({error}). "
            "It may be rebuilding the export; please try again in a few minutes."
        )
    if isinstance(error, requests.exceptions.Timeout):
        return "Request timed out. The data source may be slow. Please try again."
    if isinstance(error, requests.exceptions.ConnectionError):
        return "Connection failed. Please check your internet connection and try again."
    if isinstance(error, requests.exceptions.HTTPError):
        return f"Server error: {error}. The data source may be temporarily unavailable."
    if isinstance(error, ValueError):
        return f"Data validation error: {error}"
    return f"Unexpected error loading data: {error}"


def format_daily_rate(per_day: float) -> str:
    """Incidents per day for the KPI card.

    One decimal place, or two significant digits below one a day, so a rare
    category shows as 0.0043 rather than 0.0.
    """
    return f"{per_day:.1f}" if per_day >= 1 else f"{per_day:.2g}"


def setup_sidebar(categories: list[str], years: list[int]) -> tuple[str, int | str]:
    """Set up sidebar filters and return the selected category and year."""
    st.sidebar.header("Filters")
    selected_category = st.sidebar.selectbox("Offense Category", ["All"] + sorted(categories))
    selected_year = st.sidebar.selectbox("Year", ["All"] + sorted(years))
    return selected_category, selected_year


def overview_page() -> None:
    """Render the Overview page content."""
    df_filtered = st.session_state.get("df_filtered")
    selected_category = st.session_state.get("selected_category", "All")
    data_end_date = st.session_state.get("data_end_date")

    st.header("Overview")

    if df_filtered is None or df_filtered.empty:
        st.warning("No data available for selected filters.")
        st.stop()

    # ── KPI Metrics ─────────────────────────────────────────────────────────────
    col1, col2, col3, col4 = st.columns(4)
    # Rates are measured over the whole selected period, not from the first to
    # the last incident, so a rarely reported category is not overstated.
    period_start: date = st.session_state["period_start"]
    period_end: date = st.session_state["period_end"]
    total_days = (period_end - period_start).days + 1
    avg_daily = len(df_filtered) / total_days if total_days > 0 else 0

    incidents_delta, avg_delta = compute_yoy_deltas(df_filtered, data_end_date)

    # More crime is bad news, so increases are shown in red (inverse colors).
    with col1:
        st.metric(
            "Total Incidents", f"{len(df_filtered):,}",
            delta=incidents_delta, delta_color="inverse",
            help="Counts offenses: a report with several charges is counted once per charge.",
        )
    with col2:
        st.metric(
            "Date Range", f"{total_days:,} days",
            delta=f"{period_start} to {period_end}", delta_color="off", delta_arrow="off",
        )
    with col3:
        st.metric("Avg Daily", format_daily_rate(avg_daily), delta=avg_delta, delta_color="inverse")
    with col4:
        if selected_category != "All":
            st.metric("Selected Category", selected_category)
        else:
            st.metric("Most Common", df_filtered["offense_category"].mode()[0])

    st.divider()

    # ── Summary Charts (side-by-side) ───────────────────────────────────────────
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
        st.plotly_chart(fig_cat, width="stretch")

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
        st.plotly_chart(fig_status, width="stretch")

    # ── Daily Trend Sparkline ───────────────────────────────────────────────────
    st.subheader("Daily Incidents")
    daily = daily_incident_counts(df_filtered, period_start, period_end)

    fig_daily = px.line(daily, x="date", y="count", labels={"date": "", "count": "Incidents"})
    fig_daily.update_layout(height=200)
    apply_dark_plotly(fig_daily, BLUE_ACCENT)
    st.plotly_chart(fig_daily, width="stretch")

    # ── Footer ──────────────────────────────────────────────────────────────────
    st.markdown(
        f'<p class="footer-text">Data source: Detroit Open Data Portal · '
        f'Data through {data_end_date}</p>',
        unsafe_allow_html=True,
    )


def main() -> None:
    """Load data, populate session state for child pages, and run navigation."""
    pg = st.navigation([
        st.Page(overview_page, title="Overview", default=True),
        st.Page("pages/2_Trends.py", title="Trends"),
        st.Page("pages/3_Geography.py", title="Geography"),
        st.Page("pages/4_Analysis.py", title="Pivot Analysis"),
        st.Page("pages/5_Map.py", title="Map"),
    ])

    # The dataset is refreshed once it is DATA_CACHE_TTL old. If that refresh
    # fails, the previous copy stays in use and the failure is reported.
    try:
        df, refresh_error = dataset_cache().get(fetch_data)
    except Exception as e:
        st.error(load_error_message(e))
        return

    if refresh_error is not None:
        st.warning(
            "The data could not be refreshed, so the last copy that loaded is shown. "
            + load_error_message(refresh_error)
        )

    # Categories ordered most frequent first across the whole dataset. The Map
    # page uses this order so each category keeps its color whatever the filter.
    category_order = df["offense_category"].value_counts().index.tolist()
    years = df["incident_year"].unique().tolist()

    selected_category, selected_year = setup_sidebar(category_order, years)

    # Filter data (these produce new frames; the cached df is never modified)
    df_all_years = df
    if selected_category != "All":
        df_all_years = df_all_years[df_all_years["offense_category"] == selected_category]
    df_filtered = df_all_years
    if selected_year != "All":
        df_filtered = df_filtered[df_filtered["incident_year"] == selected_year]

    # First and last day the dataset covers. The last one is used as "today"
    # for year-to-date maths.
    data_start_date = df["incident_occurred_at"].min().date()
    data_end_date = df["incident_occurred_at"].max().date()
    period_start, period_end = selection_period(selected_year, data_start_date, data_end_date)

    # Store in session state for child pages
    st.session_state["df_filtered"] = df_filtered
    # The category selection across every year, which the forecast is fitted on
    st.session_state["df_all_years"] = df_all_years
    st.session_state["selected_category"] = selected_category
    st.session_state["selected_year"] = selected_year
    st.session_state["category_order"] = category_order
    st.session_state["data_start_date"] = data_start_date
    st.session_state["data_end_date"] = data_end_date
    # Days covered by the Year filter, which rates and daily charts span
    st.session_state["period_start"] = period_start
    st.session_state["period_end"] = period_end

    pg.run()


if __name__ == "__main__":
    main()
