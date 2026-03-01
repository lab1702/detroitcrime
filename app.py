"""Detroit Crime Incidents Dashboard

Entry point for the multi-page Streamlit application.
Handles data loading, caching, sidebar filters, session state,
and displays the Overview page (KPIs, summary charts, daily sparkline).
"""

from typing import Optional
from datetime import datetime
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests

from constants import (
    DATA_URL, DATA_CACHE_TTL,
    MIN_DATE, DETROIT_LAT_MIN, DETROIT_LAT_MAX, DETROIT_LON_MIN, DETROIT_LON_MAX,
    CHART_HEIGHT, TOP_N_ITEMS,
)
from utils import validate_data, safe_load_data_from_url, clean_and_filter_data
from theme import inject_css, apply_dark_plotly, BLUE_ACCENT, GREEN_ACCENT, RED_ACCENT

# Configure page
st.set_page_config(
    page_title="Detroit Crime Incidents",
    page_icon="\U0001f694",
    layout="wide",
)

inject_css()


@st.cache_data(ttl=DATA_CACHE_TTL, show_spinner=False)
def load_data() -> Optional[pd.DataFrame]:
    """Load Detroit crime data with error handling and validation."""
    try:
        df = safe_load_data_from_url(DATA_URL)
        validate_data(df)
        lat_bounds = (DETROIT_LAT_MIN, DETROIT_LAT_MAX)
        lon_bounds = (DETROIT_LON_MIN, DETROIT_LON_MAX)
        df = clean_and_filter_data(df, MIN_DATE, lat_bounds, lon_bounds)
        return df
    except requests.exceptions.Timeout:
        st.error("Request timed out. The data source may be slow. Please try again.")
        return None
    except requests.exceptions.ConnectionError:
        st.error("Connection failed. Please check your internet connection and try again.")
        return None
    except requests.exceptions.HTTPError as e:
        st.error(f"Server error: {str(e)}. The data source may be temporarily unavailable.")
        return None
    except ValueError as e:
        st.error(f"Data validation error: {str(e)}")
        return None
    except Exception as e:
        st.error(f"Unexpected error loading data: {str(e)}")
        return None


def setup_sidebar(df: pd.DataFrame) -> str:
    """Set up sidebar filters and return the selected category."""
    st.sidebar.header("Filters")
    offense_categories = ["All"] + sorted(df["offense_category"].unique().tolist())
    selected_category = st.sidebar.selectbox("Offense Category", offense_categories)
    return selected_category


def compute_yoy_deltas(df_filtered: pd.DataFrame):
    """Compute year-over-year deltas for Total Incidents and Avg Daily.

    Returns (incidents_delta, avg_delta) as formatted strings or None.
    """
    current_year = datetime.now().year
    prior_year = current_year - 1

    cy_data = df_filtered[df_filtered["incident_occurred_at"].dt.year == current_year]
    py_data = df_filtered[df_filtered["incident_occurred_at"].dt.year == prior_year]

    if len(cy_data) > 0 and len(py_data) > 0:
        days_elapsed = (datetime.now() - datetime(current_year, 1, 1)).days
        if days_elapsed > 60:  # Need at least 2 months of data
            cy_annualized = len(cy_data) * (365 / days_elapsed)
            py_total = len(py_data)

            incidents_delta = f"{((cy_annualized - py_total) / py_total) * 100:+.1f}%"

            # For avg daily
            cy_avg = len(cy_data) / days_elapsed
            py_days = (py_data["date"].max() - py_data["date"].min()).days + 1
            py_avg = len(py_data) / py_days if py_days > 0 else 0
            avg_delta = f"{((cy_avg - py_avg) / py_avg) * 100:+.1f}%" if py_avg > 0 else None
        else:
            incidents_delta = None
            avg_delta = None
    else:
        incidents_delta = None
        avg_delta = None

    return incidents_delta, avg_delta


def main() -> None:
    """Load data, populate session state for child pages, and render Overview."""
    df = load_data()
    if df is None:
        st.error("Failed to load data. Please try again later.")
        return

    selected_category = setup_sidebar(df)

    # Filter data
    if selected_category == "All":
        df_filtered = df
    else:
        df_filtered = df[df["offense_category"] == selected_category]

    # Store in session state for child pages
    st.session_state["df"] = df
    st.session_state["df_filtered"] = df_filtered
    st.session_state["selected_category"] = selected_category

    # ── Overview page content ───────────────────────────────────────────────────
    st.header("Overview")

    if df_filtered.empty:
        st.warning("No data available for selected filters.")
        st.stop()

    # ── KPI Metrics ─────────────────────────────────────────────────────────────
    col1, col2, col3, col4 = st.columns(4)
    min_date = df_filtered["date"].min()
    max_date = df_filtered["date"].max()
    total_days = (max_date - min_date).days + 1
    avg_daily = len(df_filtered) / total_days if total_days > 0 else 0

    incidents_delta, avg_delta = compute_yoy_deltas(df_filtered)

    with col1:
        st.metric("Total Incidents", f"{len(df_filtered):,}", delta=incidents_delta)
    with col2:
        st.metric("Date Range", f"{total_days:,} days", f"{min_date} to {max_date}")
    with col3:
        st.metric("Avg Daily", f"{avg_daily:.1f}", delta=avg_delta)
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

    # ── Daily Trend Sparkline ───────────────────────────────────────────────────
    st.subheader("Daily Incidents")
    daily = df_filtered.groupby("date").size().reset_index(name="count")
    daily["date"] = pd.to_datetime(daily["date"])
    date_range = pd.date_range(start=daily["date"].min(), end=daily["date"].max(), freq="D")
    daily = pd.DataFrame({"date": date_range}).merge(daily, on="date", how="left").fillna(0)

    fig_daily = px.line(daily, x="date", y="count", labels={"date": "", "count": "Incidents"})
    fig_daily.update_layout(height=200)
    apply_dark_plotly(fig_daily, BLUE_ACCENT)
    st.plotly_chart(fig_daily, use_container_width=True)

    # ── Footer ──────────────────────────────────────────────────────────────────
    st.markdown(
        f'<p class="footer-text">Data source: Detroit Open Data Portal \u00b7 '
        f'Last updated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>',
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
