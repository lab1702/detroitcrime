# Dashboard Redesign Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Restructure the Detroit Crime dashboard from a single-file 12-tab app into a 4-page multi-page Streamlit app with a dark command-center theme.

**Architecture:** Convert `app.py` into an entry point that loads/caches data into `st.session_state`, then create 4 page files under `pages/`. Add a `.streamlit/config.toml` for dark theme. Inject custom CSS for KPI card styling. Apply `plotly_dark` template to all charts with contextual accent colors.

**Tech Stack:** Streamlit (multi-page), Plotly (dark template), Prophet, custom CSS via `st.markdown`

---

### Task 1: Create Dark Theme Config

**Files:**
- Create: `.streamlit/config.toml`

**Step 1: Create the Streamlit config directory and theme file**

```toml
[theme]
primaryColor = "#00d4ff"
backgroundColor = "#0E1117"
secondaryBackgroundColor = "#1a1f2e"
textColor = "#e0e0e0"
font = "sans serif"
```

**Step 2: Verify the file exists**

Run: `cat .streamlit/config.toml`
Expected: Contents of the theme file shown.

**Step 3: Commit**

```bash
git add .streamlit/config.toml
git commit -m "feat: add dark theme config"
```

---

### Task 2: Create Shared Theme Module

**Files:**
- Create: `theme.py`

This module provides: (1) CSS for KPI cards, page titles, and sidebar styling, (2) a function to apply `plotly_dark` template with accent color overrides to any Plotly figure, (3) color constants for the 3 accent palettes (blue/cyan for trends, red/amber for geography, green/teal for positive metrics).

**Step 1: Create `theme.py`**

```python
"""Theme and styling for Detroit Crime Dashboard"""

import plotly.graph_objects as go
import streamlit as st

# Accent color palettes
BLUE_ACCENT = "#00d4ff"      # Trends, temporal data
RED_ACCENT = "#ff4b4b"       # Geographic hotspots, severity
AMBER_ACCENT = "#ffaa00"     # Secondary geographic
GREEN_ACCENT = "#00c853"     # Positive metrics (clearances)
TEAL_ACCENT = "#00bfa5"      # Secondary positive

# Dark backgrounds
CARD_BG = "#1a1f2e"
CARD_BORDER = "#2d3548"

CUSTOM_CSS = """
<style>
    /* KPI metric cards */
    div[data-testid="stMetric"] {
        background-color: #1a1f2e;
        border: 1px solid #2d3548;
        border-radius: 8px;
        padding: 16px 20px;
        box-shadow: 0 0 15px rgba(0, 212, 255, 0.05);
    }
    div[data-testid="stMetric"] label {
        color: #8899aa;
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    div[data-testid="stMetric"] [data-testid="stMetricValue"] {
        color: #e0e0e0;
        font-size: 1.8rem;
        font-weight: 700;
    }

    /* Page headers */
    h1 {
        color: #e0e0e0 !important;
        font-weight: 700 !important;
        letter-spacing: -0.02em !important;
    }
    h2, h3 {
        color: #c0c8d4 !important;
        font-weight: 600 !important;
    }

    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background-color: #0a0e14;
        border-right: 1px solid #1a1f2e;
    }

    /* Footer */
    .footer-text {
        color: #556677;
        font-size: 0.8rem;
        text-align: center;
        padding-top: 2rem;
        border-top: 1px solid #1a1f2e;
    }
</style>
"""


def inject_css():
    """Inject custom CSS into the page."""
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def apply_dark_plotly(fig: go.Figure, accent_color: str = BLUE_ACCENT) -> go.Figure:
    """Apply dark theme to a Plotly figure with an accent color.

    Args:
        fig: Plotly figure to style.
        accent_color: Primary accent color for the chart.

    Returns:
        The same figure with dark theme applied.
    """
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0E1117",
        plot_bgcolor="#0E1117",
        font=dict(color="#c0c8d4"),
        xaxis=dict(gridcolor="#1a1f2e", zerolinecolor="#1a1f2e"),
        yaxis=dict(gridcolor="#1a1f2e", zerolinecolor="#1a1f2e"),
        coloraxis_colorbar=dict(
            tickfont=dict(color="#c0c8d4"),
            titlefont=dict(color="#c0c8d4"),
        ),
        margin=dict(l=20, r=20, t=40, b=20),
    )
    # Apply accent color to bar traces
    for trace in fig.data:
        if hasattr(trace, "marker") and trace.type == "bar":
            if trace.marker.color is None:
                trace.marker.color = accent_color
    return fig
```

**Step 2: Verify syntax**

Run: `python -c "import theme; print('OK')"`
Expected: `OK`

**Step 3: Commit**

```bash
git add theme.py
git commit -m "feat: add theme module with dark styling and accent colors"
```

---

### Task 3: Refactor `app.py` into Multi-Page Entry Point

**Files:**
- Modify: `app.py` (rewrite to entry point only)

The new `app.py` keeps: page config, data loading, sidebar filter, and stores everything in `st.session_state`. It removes: all chart functions, all tab rendering, metrics display, footer. It adds: CSS injection, a welcome message when no page is selected (though Streamlit auto-navigates to the first page).

**Step 1: Rewrite `app.py`**

```python
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
```

**Step 2: Verify the app still starts**

Run: `timeout 10 streamlit run app.py --server.headless true 2>&1 | head -5`
Expected: Streamlit starts without import errors.

**Step 3: Commit**

```bash
git add app.py
git commit -m "refactor: convert app.py to multi-page entry point"
```

---

### Task 4: Create Overview Page

**Files:**
- Create: `pages/1_Overview.py`

This is the landing page. Shows 4 KPI metric cards, then categories + case status side-by-side, then a compact daily trend. Uses `st.session_state` to get filtered data.

**Step 1: Create the pages directory and Overview page**

```python
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
```

**Step 2: Verify the page renders without errors**

Run: `python -c "import pages; print('dir ok')" 2>&1 || echo "expected - pages is a streamlit dir not a package"`

Verify file exists: `ls pages/1_Overview.py`

**Step 3: Commit**

```bash
git add pages/1_Overview.py
git commit -m "feat: add Overview page with KPI cards and summary charts"
```

---

### Task 5: Create Trends Page

**Files:**
- Create: `pages/2_Trends.py`

Contains: daily incidents (full width), year-over-year + heatmap (side-by-side), forecast (full width). All charts use blue/cyan accents. Reuses the chart creation logic from the old `app.py` but applies dark styling.

**Step 1: Create the Trends page**

```python
"""Trends page — temporal analysis of crime incidents."""

from typing import Optional
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from prophet import Prophet
from prophet.make_holidays import make_holidays_df

from constants import (
    CHART_HEIGHT, MIN_FORECAST_DAYS, FORECAST_PERIODS, HISTORICAL_MONTHS,
    MIN_PROJECTION_DAYS, FORECAST_CACHE_TTL, YOY_CACHE_TTL,
)
from theme import inject_css, apply_dark_plotly, BLUE_ACCENT, TEAL_ACCENT

inject_css()

st.header("Trends")

if "df_filtered" not in st.session_state:
    st.warning("Loading data — please wait…")
    st.stop()

df_filtered: pd.DataFrame = st.session_state["df_filtered"]

if df_filtered.empty:
    st.warning("No data available for selected filters.")
    st.stop()


# ── Daily Incidents (full width) ─────────────────────────────────────────────
st.subheader("Daily Incidents")
daily = df_filtered.groupby("date").size().reset_index(name="count")
daily["date"] = pd.to_datetime(daily["date"])
date_range = pd.date_range(start=daily["date"].min(), end=daily["date"].max(), freq="D")
daily = pd.DataFrame({"date": date_range}).merge(daily, on="date", how="left").fillna(0)

fig_daily = px.line(daily, x="date", y="count", labels={"date": "Date", "count": "Incidents"})
fig_daily.update_layout(height=CHART_HEIGHT)
apply_dark_plotly(fig_daily, BLUE_ACCENT)
st.plotly_chart(fig_daily, use_container_width=True)


# ── Year over Year + Heatmap (side-by-side) ──────────────────────────────────
left, right = st.columns(2)

with left:
    st.subheader("Year over Year")

    @st.cache_data(ttl=YOY_CACHE_TTL)
    def create_yoy(df):
        df = df.copy()
        df["year"] = df["incident_occurred_at"].dt.year
        yearly = df.groupby("year").size().reset_index(name="count")
        current_year = datetime.now().year
        current_year_data = df[df["year"] == current_year]

        fig = px.bar(yearly, x="year", y="count", labels={"year": "Year", "count": "Incidents"})

        if len(current_year_data) > 0:
            start_of_year = datetime(current_year, 1, 1)
            days_elapsed = (datetime.now() - start_of_year).days
            if days_elapsed >= MIN_PROJECTION_DAYS:
                incidents_so_far = len(current_year_data)
                days_in_year = 366 if current_year % 4 == 0 else 365
                projected = incidents_so_far * (days_in_year / days_elapsed)
                current_count = (
                    yearly[yearly["year"] == current_year]["count"].iloc[0]
                    if current_year in yearly["year"].values
                    else 0
                )
                projection_height = projected - current_count
                if projection_height > 0:
                    fig.add_bar(
                        x=[current_year],
                        y=[projection_height],
                        base=[current_count],
                        name="Projected",
                        marker_color="rgba(255, 165, 0, 0.3)",
                        showlegend=True,
                    )

        fig.update_layout(height=CHART_HEIGHT, barmode="overlay")
        return fig

    fig_yoy = create_yoy(df_filtered)
    apply_dark_plotly(fig_yoy, BLUE_ACCENT)
    st.plotly_chart(fig_yoy, use_container_width=True)

with right:
    st.subheader("Hour of Day × Day of Week")

    @st.cache_data(ttl=3600)
    def create_heatmap(df):
        heatmap_data = (
            df.groupby(["incident_day_of_week", "incident_hour_of_day"])
            .size()
            .reset_index(name="count")
        )
        pivot = heatmap_data.pivot(
            index="incident_day_of_week", columns="incident_hour_of_day", values="count"
        ).fillna(0)

        day_labels = [
            "Sunday", "Monday", "Tuesday", "Wednesday",
            "Thursday", "Friday", "Saturday",
        ]

        fig = px.imshow(
            pivot,
            labels=dict(x="Hour", y="Day", color="Incidents"),
            aspect="auto",
            color_continuous_scale="RdYlGn_r",
        )
        fig.update_yaxes(tickvals=list(range(1, 8)), ticktext=day_labels)
        fig.update_layout(height=CHART_HEIGHT)
        return fig

    fig_hm = create_heatmap(df_filtered)
    apply_dark_plotly(fig_hm, BLUE_ACCENT)
    st.plotly_chart(fig_hm, use_container_width=True)


# ── Forecast (full width) ────────────────────────────────────────────────────
st.subheader("1-Year Forecast")


@st.cache_data(ttl=FORECAST_CACHE_TTL)
def create_forecast(df: pd.DataFrame) -> Optional[go.Figure]:
    """Create Prophet forecast for incident count."""
    try:
        daily_counts = df.groupby("date").size().reset_index()
        daily_counts.columns = ["ds", "y"]
        daily_counts["ds"] = pd.to_datetime(daily_counts["ds"])

        date_range = pd.date_range(
            start=daily_counts["ds"].min(), end=daily_counts["ds"].max(), freq="D"
        )
        daily_counts = (
            pd.DataFrame({"ds": date_range})
            .merge(daily_counts, on="ds", how="left")
            .fillna(0)
        )

        if len(daily_counts) < MIN_FORECAST_DAYS:
            return None

        holidays = make_holidays_df(
            year_list=list(
                range(daily_counts["ds"].dt.year.min(), daily_counts["ds"].dt.year.max() + 2)
            ),
            country="US",
        )

        model = Prophet(holidays=holidays)
        model.fit(daily_counts)
        future = model.make_future_dataframe(periods=FORECAST_PERIODS)
        forecast = model.predict(future)

        fig = go.Figure()

        forecast_start = daily_counts["ds"].max()
        three_months_before = forecast_start - pd.DateOffset(months=HISTORICAL_MONTHS)
        historical = daily_counts[daily_counts["ds"] >= three_months_before]

        fig.add_trace(
            go.Scatter(
                x=historical["ds"],
                y=historical["y"],
                mode="markers",
                name="Historical",
                marker=dict(color=BLUE_ACCENT, size=4),
            )
        )

        future_dates = forecast[forecast["ds"] > daily_counts["ds"].max()]
        fig.add_trace(
            go.Scatter(
                x=future_dates["ds"],
                y=future_dates["yhat"],
                mode="lines",
                name="Forecast",
                line=dict(color="#ff4b4b", width=2),
            )
        )

        fig.add_trace(
            go.Scatter(
                x=future_dates["ds"],
                y=future_dates["yhat_upper"],
                fill=None,
                mode="lines",
                line_color="rgba(0,0,0,0)",
                showlegend=False,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=future_dates["ds"],
                y=future_dates["yhat_lower"],
                fill="tonexty",
                mode="lines",
                line_color="rgba(0,0,0,0)",
                name="Confidence",
                fillcolor="rgba(255,75,75,0.15)",
            )
        )

        # Holiday annotations
        forecast_holidays = holidays[
            (holidays["ds"] >= forecast_start) & (holidays["ds"] <= future_dates["ds"].max())
        ]
        if not forecast_holidays.empty:
            y_max = future_dates["yhat"].max()
            y_min = future_dates["yhat"].min()
            y_range = y_max - y_min
            forecast_holidays = forecast_holidays.sort_values("ds")

            for i, (_, holiday) in enumerate(forecast_holidays.iterrows()):
                fig.add_vline(
                    x=holiday["ds"],
                    line_dash="dash",
                    line_color="rgba(0, 200, 83, 0.4)",
                    line_width=1,
                )
                level = i % 4
                y_pos = y_max - (y_range * 0.1) - level * (y_range * 0.25)
                name = holiday["holiday"]
                display = name if len(name) <= 12 else name[:9] + "..."
                fig.add_annotation(
                    x=holiday["ds"],
                    y=y_pos,
                    text=display,
                    showarrow=True,
                    arrowhead=1,
                    arrowsize=0.7,
                    arrowwidth=1,
                    arrowcolor="rgba(0, 200, 83, 0.6)",
                    ax=0,
                    ay=-25 - level * 25,
                    bgcolor="rgba(26, 31, 46, 0.95)",
                    bordercolor="rgba(0, 200, 83, 0.6)",
                    borderwidth=1,
                    font=dict(size=8, color="#c0c8d4"),
                )

        fig.update_layout(
            xaxis_title="Date",
            yaxis_title="Incidents",
            height=CHART_HEIGHT,
        )
        return fig
    except Exception as e:
        st.error(f"Error creating forecast: {str(e)}")
        return None


forecast_fig = create_forecast(df_filtered)
if forecast_fig:
    apply_dark_plotly(forecast_fig, BLUE_ACCENT)
    st.plotly_chart(forecast_fig, use_container_width=True)
else:
    st.info("Not enough data for forecasting (minimum 30 days required).")
```

**Step 2: Verify file exists**

Run: `ls pages/2_Trends.py`

**Step 3: Commit**

```bash
git add pages/2_Trends.py
git commit -m "feat: add Trends page with daily, YoY, heatmap, and forecast"
```

---

### Task 6: Create Geography Page

**Files:**
- Create: `pages/3_Geography.py`

Contains: neighborhoods + precincts (side-by-side), districts + zip codes (side-by-side), locations (full width). Red/amber accents.

**Step 1: Create the Geography page**

```python
"""Geography page — location-based crime analysis."""

import streamlit as st
import pandas as pd
import plotly.express as px

from constants import CHART_HEIGHT, TOP_N_ITEMS
from utils import format_categorical_column, format_zip_code_column
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
```

**Step 2: Verify file exists**

Run: `ls pages/3_Geography.py`

**Step 3: Commit**

```bash
git add pages/3_Geography.py
git commit -m "feat: add Geography page with neighborhoods, precincts, districts, zips, locations"
```

---

### Task 7: Create Analysis Page

**Files:**
- Create: `pages/4_Analysis.py`

Contains: the pivot table with controls, heatmap, and data table. Moved from the old tab12 logic.

**Step 1: Create the Analysis page**

```python
"""Analysis page — interactive pivot table."""

import streamlit as st
import pandas as pd
import plotly.express as px

from constants import CHART_HEIGHT, PIVOT_COLUMNS, COLOR_SCALE_HEATMAP, CATEGORICAL_VARS
from utils import format_categorical_column, format_zip_code_column
from theme import inject_css, apply_dark_plotly, TEAL_ACCENT

inject_css()

st.header("Pivot Analysis")

if "df_filtered" not in st.session_state:
    st.warning("Loading data — please wait…")
    st.stop()

df_filtered: pd.DataFrame = st.session_state["df_filtered"]

if df_filtered.empty:
    st.warning("No data available for selected filters.")
    st.stop()

# ── Pivot controls ───────────────────────────────────────────────────────────
col1, col2, col3 = st.columns([2, 2, 1])

if "pivot_rows_selection" not in st.session_state:
    st.session_state.pivot_rows_selection = PIVOT_COLUMNS[0]
if "pivot_columns_selection" not in st.session_state:
    st.session_state.pivot_columns_selection = PIVOT_COLUMNS[1]

with col1:
    rows = st.selectbox(
        "Rows (Group by):",
        PIVOT_COLUMNS,
        index=(
            PIVOT_COLUMNS.index(st.session_state.pivot_rows_selection)
            if st.session_state.pivot_rows_selection in PIVOT_COLUMNS
            else 0
        ),
        key="pivot_rows",
    )
    st.session_state.pivot_rows_selection = rows

with col2:
    columns = st.selectbox(
        "Columns:",
        PIVOT_COLUMNS,
        index=(
            PIVOT_COLUMNS.index(st.session_state.pivot_columns_selection)
            if st.session_state.pivot_columns_selection in PIVOT_COLUMNS
            else 1
        ),
        key="pivot_columns",
    )
    st.session_state.pivot_columns_selection = columns

with col3:
    st.write("")
    if st.button("Swap", help="Swap rows and columns", key="swap_pivot"):
        temp = st.session_state.pivot_rows_selection
        st.session_state.pivot_rows_selection = st.session_state.pivot_columns_selection
        st.session_state.pivot_columns_selection = temp

# ── Create pivot table ───────────────────────────────────────────────────────
try:
    required_cols = list(set([rows, columns, "incident_entry_id"]))
    df_sub = df_filtered[required_cols].copy()

    if rows in CATEGORICAL_VARS:
        prefix = "Precinct" if rows == "police_precinct" else "District" if rows == "council_district" else ""
        if rows == "zip_code":
            df_sub = format_zip_code_column(df_sub, rows)
        else:
            df_sub = format_categorical_column(df_sub, rows, prefix)

    if columns in CATEGORICAL_VARS and columns != rows:
        prefix = "Precinct" if columns == "police_precinct" else "District" if columns == "council_district" else ""
        if columns == "zip_code":
            df_sub = format_zip_code_column(df_sub, columns)
        else:
            df_sub = format_categorical_column(df_sub, columns, prefix)

    pivot_table = pd.pivot_table(
        df_sub, index=rows, columns=columns, values="incident_entry_id",
        aggfunc="count", fill_value=0,
    )

    if rows == "zip_code":
        pivot_table.index = pivot_table.index.astype(str)
    if columns == "zip_code":
        pivot_table.columns = pivot_table.columns.astype(str)

    fig = px.imshow(
        pivot_table,
        labels=dict(x=columns, y=rows, color="Count"),
        aspect="auto",
        color_continuous_scale=COLOR_SCALE_HEATMAP,
    )
    if rows == "zip_code":
        fig.update_yaxes(type="category")
    if columns == "zip_code":
        fig.update_xaxes(type="category")
    fig.update_layout(height=CHART_HEIGHT)
    apply_dark_plotly(fig, TEAL_ACCENT)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Pivot Table Data")
    st.info(f"Table size: {pivot_table.shape[0]} rows × {pivot_table.shape[1]} columns")
    st.dataframe(pivot_table, use_container_width=True)

except Exception as e:
    st.error(f"Error creating pivot table: {str(e)}")
```

**Step 2: Verify file exists**

Run: `ls pages/4_Analysis.py`

**Step 3: Commit**

```bash
git add pages/4_Analysis.py
git commit -m "feat: add Analysis page with pivot table"
```

---

### Task 8: Verify Full Application

**Files:**
- None (verification only)

**Step 1: Check that all pages exist**

Run: `ls -la pages/`
Expected: 4 files (1_Overview.py, 2_Trends.py, 3_Geography.py, 4_Analysis.py)

**Step 2: Check that imports work for all pages**

Run: `python -c "from theme import inject_css, apply_dark_plotly; print('theme OK')"`
Run: `python -c "from constants import CHART_HEIGHT, PIVOT_COLUMNS; print('constants OK')"`
Run: `python -c "from utils import validate_data; print('utils OK')"`
Expected: All print OK.

**Step 3: Start the app and verify it loads**

Run: `timeout 15 streamlit run app.py --server.headless true 2>&1 | head -10`
Expected: Streamlit starts without errors, shows URL.

**Step 4: Final commit if any fixups needed**

Only commit if changes were required during verification.

---

### Task 9: Clean Up and Final Commit

**Files:**
- Possibly modify: any files that needed fixups

**Step 1: Run git status to check for loose files**

Run: `git status`
Expected: Clean working tree or only expected untracked files.

**Step 2: Verify all pages load without import errors**

Run: `python -c "exec(open('pages/1_Overview.py').read())" 2>&1 | head -3`

Note: This will fail because Streamlit context isn't available outside `streamlit run`, but it should fail on `st.header()` or `st.session_state`, NOT on import errors. If it fails on imports, fix those.

**Step 3: Final commit with any remaining changes**

```bash
git add -A
git commit -m "chore: cleanup after dashboard redesign"
```
