"""Detroit Crime Incidents Dashboard

Entry point for the multi-page Streamlit application.
Handles data loading, caching, sidebar filters, session state,
and displays the Overview page (KPIs, summary charts, daily sparkline).
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import requests

from constants import (
    DATA_URL, DATA_CACHE_TTL, REQUIRED_COLUMNS,
    MIN_DATE, DETROIT_LAT_MIN, DETROIT_LAT_MAX, DETROIT_LON_MIN, DETROIT_LON_MAX,
    CHART_HEIGHT, TOP_N_ITEMS,
)
from utils import (
    validate_data, safe_load_data_from_url, clean_and_filter_data, compute_yoy_deltas,
)
from theme import inject_css, apply_dark_plotly, BLUE_ACCENT, GREEN_ACCENT, RED_ACCENT

# Configure page
st.set_page_config(
    page_title="Detroit Crime Incidents",
    page_icon="\U0001f694",
    layout="wide",
)

inject_css()


@st.cache_resource(ttl=DATA_CACHE_TTL, show_spinner=False)
def load_data() -> pd.DataFrame:
    """Load Detroit crime data. Raises on failure so errors are not cached.

    Cached as a resource, so every session and rerun shares this one DataFrame
    instead of unpickling its own copy. Treat it as read-only: filter or copy
    it before changing anything.
    """
    df = safe_load_data_from_url(DATA_URL, usecols=REQUIRED_COLUMNS)
    validate_data(df)
    lat_bounds = (DETROIT_LAT_MIN, DETROIT_LAT_MAX)
    lon_bounds = (DETROIT_LON_MIN, DETROIT_LON_MAX)
    return clean_and_filter_data(df, MIN_DATE, lat_bounds, lon_bounds)


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
    min_date = df_filtered["date"].min()
    max_date = df_filtered["date"].max()
    total_days = (max_date - min_date).days + 1
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
            delta=f"{min_date} to {max_date}", delta_color="off", delta_arrow="off",
        )
    with col3:
        st.metric("Avg Daily", f"{avg_daily:.1f}", delta=avg_delta, delta_color="inverse")
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
    daily = df_filtered.groupby("date").size().reset_index(name="count")
    daily["date"] = pd.to_datetime(daily["date"])
    date_range = pd.date_range(start=daily["date"].min(), end=daily["date"].max(), freq="D")
    daily = pd.DataFrame({"date": date_range}).merge(daily, on="date", how="left").fillna(0)

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

    try:
        df = load_data()
    except requests.exceptions.Timeout:
        st.error("Request timed out. The data source may be slow. Please try again.")
        return
    except requests.exceptions.ConnectionError:
        st.error("Connection failed. Please check your internet connection and try again.")
        return
    except requests.exceptions.HTTPError as e:
        st.error(f"Server error: {str(e)}. The data source may be temporarily unavailable.")
        return
    except ValueError as e:
        st.error(f"Data validation error: {str(e)}")
        return
    except Exception as e:
        st.error(f"Unexpected error loading data: {str(e)}")
        return

    # Categories ordered most frequent first across the whole dataset. The Map
    # page uses this order so each category keeps its color whatever the filter.
    category_order = df["offense_category"].value_counts().index.tolist()
    years = df["incident_year"].unique().tolist()

    selected_category, selected_year = setup_sidebar(category_order, years)

    # Filter data (these produce new frames; the cached df is never modified)
    df_filtered = df
    if selected_category != "All":
        df_filtered = df_filtered[df_filtered["offense_category"] == selected_category]
    if selected_year != "All":
        df_filtered = df_filtered[df_filtered["incident_year"] == selected_year]

    # Store in session state for child pages
    st.session_state["df_filtered"] = df_filtered
    st.session_state["selected_category"] = selected_category
    st.session_state["category_order"] = category_order
    # Last day the dataset covers, used as "today" for year-to-date maths
    st.session_state["data_end_date"] = df["incident_occurred_at"].max().date()

    pg.run()


if __name__ == "__main__":
    main()
