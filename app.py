"""Detroit Crime Incidents Dashboard

Entry point for the multi-page Streamlit application.
Handles data loading, caching, sidebar filters, and session state.
"""

from typing import Optional
import streamlit as st
import pandas as pd
import requests

from constants import (
    DATA_URL, DATA_CACHE_TTL,
    MIN_DATE, DETROIT_LAT_MIN, DETROIT_LAT_MAX, DETROIT_LON_MIN, DETROIT_LON_MAX,
)
from utils import validate_data, safe_load_data_from_url, clean_and_filter_data
from theme import inject_css

# Configure page
st.set_page_config(
    page_title="Detroit Crime Incidents",
    page_icon="🚔",
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


def main() -> None:
    """Load data and populate session state for child pages."""
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

    # Landing content (shown on the main page)
    st.title("🚔 Detroit Crime Incidents")
    st.caption("Select a page from the sidebar to explore the data.")


if __name__ == "__main__":
    main()
