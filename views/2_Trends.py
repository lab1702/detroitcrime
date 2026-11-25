"""Trends page — temporal analysis of crime incidents."""

from typing import Optional
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from prophet import Prophet
from prophet.make_holidays import make_holidays_df

from constants import (
    CHART_HEIGHT, MIN_FORECAST_DAYS, FORECAST_PERIODS, HISTORICAL_MONTHS,
    FORECAST_CACHE_TTL, YOY_CACHE_TTL, DAY_LABELS, COLOR_SCALE_HEATMAP,
)
from utils import daily_incident_counts, project_full_year
from theme import inject_css, apply_dark_plotly, BLUE_ACCENT

inject_css()

st.header("Trends")

if "df_filtered" not in st.session_state:
    st.warning("Loading data — please wait…")
    st.stop()

df_filtered: pd.DataFrame = st.session_state["df_filtered"]

if df_filtered.empty:
    st.warning("No data available for selected filters.")
    st.stop()

# First and last day the dataset covers; the last is "today" for year-to-date
# projections and the day the forecast starts from
data_start_date = st.session_state["data_start_date"]
data_end_date = st.session_state["data_end_date"]
# Days covered by the Year filter
period_start = st.session_state["period_start"]
period_end = st.session_state["period_end"]
selected_year = st.session_state.get("selected_year", "All")
# The category selection across every year, which the forecast is fitted on
df_all_years: pd.DataFrame = st.session_state["df_all_years"]


# ── Daily Incidents (full width) ─────────────────────────────────────────────
st.subheader("Daily Incidents")
daily = daily_incident_counts(df_filtered, period_start, period_end)

fig_daily = px.line(daily, x="date", y="count", labels={"date": "Date", "count": "Incidents"})
fig_daily.update_layout(height=CHART_HEIGHT)
apply_dark_plotly(fig_daily, BLUE_ACCENT)
st.plotly_chart(fig_daily, width="stretch")


# ── Year over Year + Heatmap (side-by-side) ──────────────────────────────────
left, right = st.columns(2)

with left:
    st.subheader("Year over Year")

    @st.cache_data(ttl=YOY_CACHE_TTL)
    def create_yoy(df, as_of):
        yearly = (
            df.groupby("incident_year").size().reset_index(name="count")
            .rename(columns={"incident_year": "year"})
        )

        fig = px.bar(yearly, x="year", y="count", labels={"year": "Year", "count": "Incidents"})

        # Extend the latest year's bar to its projected full-year total
        current_year = as_of.year
        current = yearly.loc[yearly["year"] == current_year, "count"]
        if not current.empty:
            current_count = int(current.iloc[0])
            projected = project_full_year(current_count, as_of)
            if projected is not None and projected > current_count:
                fig.add_bar(
                    x=[current_year],
                    y=[projected - current_count],
                    base=[current_count],
                    name="Projected",
                    marker_color="rgba(255, 165, 0, 0.3)",
                    showlegend=True,
                )

        fig.update_layout(height=CHART_HEIGHT, barmode="overlay")
        # One tick per year: with few bars the axis would otherwise show 2024.5
        fig.update_xaxes(dtick=1)
        return fig

    fig_yoy = create_yoy(df_filtered, data_end_date)
    apply_dark_plotly(fig_yoy, BLUE_ACCENT)
    st.plotly_chart(fig_yoy, width="stretch")

with right:
    st.subheader("Hour of Day × Day of Week")

    @st.cache_data(ttl=FORECAST_CACHE_TTL)
    def create_heatmap(df):
        heatmap_data = (
            df.groupby(["incident_day_of_week", "incident_hour_of_day"])
            .size()
            .reset_index(name="count")
        )
        pivot = heatmap_data.pivot(
            index="incident_day_of_week", columns="incident_hour_of_day", values="count"
        )
        # Always the full week by 24 hours, so a small selection is not drawn
        # as a few stretched cells
        pivot = pivot.reindex(index=range(1, 8), columns=range(24)).fillna(0)

        fig = px.imshow(
            pivot,
            labels=dict(x="Hour", y="Day", color="Incidents"),
            aspect="auto",
            color_continuous_scale=COLOR_SCALE_HEATMAP,
        )
        # incident_day_of_week is ISO-numbered: 1=Monday ... 7=Sunday
        fig.update_yaxes(tickvals=list(range(1, 8)), ticktext=DAY_LABELS)
        fig.update_layout(height=CHART_HEIGHT)
        return fig

    fig_hm = create_heatmap(df_filtered)
    apply_dark_plotly(fig_hm, BLUE_ACCENT)
    st.plotly_chart(fig_hm, width="stretch")


# ── Forecast (full width) ────────────────────────────────────────────────────
st.subheader("1-Year Forecast")


@st.cache_data(ttl=FORECAST_CACHE_TTL)
def create_forecast(df: pd.DataFrame, start, end) -> Optional[go.Figure]:
    """Create Prophet forecast for incident count.

    The model is fitted on every day from `start` to `end` (the days the
    dataset covers), counting days without incidents as zero, and forecasts
    the year after `end`.

    Returns None when there is too little history to forecast. Errors are
    raised, not swallowed, so a failed fit is never cached as a result.
    """
    daily_counts = daily_incident_counts(df, start, end).rename(
        columns={"date": "ds", "count": "y"}
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

    future_dates = forecast[forecast["ds"] > forecast_start].copy()
    # Prophet knows nothing about counts: keep the forecast and its band at or above zero
    bands = ["yhat", "yhat_lower", "yhat_upper"]
    future_dates[bands] = future_dates[bands].clip(lower=0)
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


if selected_year != "All":
    st.caption(
        "The forecast is fitted on every year of data for this selection and starts "
        "after the last day of data, so the Year filter does not apply to it."
    )

try:
    forecast_fig = create_forecast(df_all_years, data_start_date, data_end_date)
except Exception as e:
    st.error(f"Error creating forecast: {str(e)}")
else:
    if forecast_fig is None:
        st.info(f"Not enough data for forecasting (minimum {MIN_FORECAST_DAYS} days required).")
    else:
        apply_dark_plotly(forecast_fig, BLUE_ACCENT)
        st.plotly_chart(forecast_fig, width="stretch")
