"""Trends page — temporal analysis of crime incidents."""

import logging
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
from utils import daily_incident_counts, holiday_label, part_year_note, partial_years, project_full_year
from theme import inject_css, apply_dark_plotly, BLUE_ACCENT

# Named here: a page script's own `__name__` says nothing about which page it is
logger = logging.getLogger("views.trends")

# How solid the bar of a year the data covers only in part is drawn (1 is a whole year)
PART_YEAR_OPACITY = 0.45


@st.cache_data(ttl=YOY_CACHE_TTL)
def create_yoy(df, start, as_of):
    """Incidents per year, for a dataset that covers `start` to `as_of`."""
    yearly = (
        df.groupby("incident_year").size().reset_index(name="count")
        .rename(columns={"incident_year": "year"})
    )

    fig = px.bar(yearly, x="year", y="count", labels={"year": "Year", "count": "Incidents"})

    # A year the data covers only in part is drawn lighter (the caption
    # under the chart says which part). Its bar would otherwise be read as
    # a fall: the current year always is one, and has no projection in its
    # first weeks.
    part_years = partial_years(start, as_of)
    fig.update_traces(
        marker_opacity=[PART_YEAR_OPACITY if year in part_years else 1 for year in yearly["year"]]
    )

    # Extend the latest year's bar to its projected full-year total
    current_year = as_of.year
    current = yearly.loc[yearly["year"] == current_year, "count"]
    if not current.empty:
        current_count = int(current.iloc[0])
        projected = project_full_year(current_count, as_of, start)
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


@st.cache_data(ttl=FORECAST_CACHE_TTL)
def create_heatmap(df):
    """The chart, and how many incidents it has no cell for.

    Those are the ones whose day or hour is missing, or is not a day of
    the week or an hour of the day.
    """
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
    left_out = len(df) - int(pivot.to_numpy().sum())

    fig = px.imshow(
        pivot,
        labels=dict(x="Hour", y="Day", color="Incidents"),
        aspect="auto",
        color_continuous_scale=COLOR_SCALE_HEATMAP,
    )
    # incident_day_of_week is ISO-numbered: 1=Monday ... 7=Sunday
    fig.update_yaxes(tickvals=list(range(1, 8)), ticktext=DAY_LABELS)
    fig.update_layout(height=CHART_HEIGHT)
    return fig, left_out


@st.cache_data(ttl=FORECAST_CACHE_TTL)
def create_forecast(df: pd.DataFrame, start, end) -> Optional[go.Figure]:
    """Create Prophet forecast for incident count.

    The model is fitted on every day from `start` to `end` (the days the
    dataset covers), counting days without incidents as zero, and forecasts
    the year after `end`.

    Returns None when there is too little history to forecast: fewer than
    MIN_FORECAST_DAYS days with an incident. Errors are raised, not
    swallowed, so a failed fit is never cached as a result.
    """
    daily_counts = daily_incident_counts(df, start, end).rename(
        columns={"date": "ds", "count": "y"}
    )

    # Days that had incidents, not rows: there is a row for every day the
    # dataset covers, however few of them the selection has anything on
    if (daily_counts["y"] > 0).sum() < MIN_FORECAST_DAYS:
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

    # Holiday annotations, for the days forecast: one on the last day of data
    # is history, like the others in the months of history shown, which have
    # no mark. Each holiday is marked once, on the day itself: the day it is
    # observed on, when that is another one, is next to it and would put a
    # second mark and name over the first. (The model is fitted on both.)
    forecast_holidays = holidays[
        (holidays["ds"] > forecast_start) & (holidays["ds"] <= future_dates["ds"].max())
        & ~holidays["holiday"].str.endswith("(observed)")
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
            fig.add_annotation(
                x=holiday["ds"],
                y=y_pos,
                text=holiday_label(holiday["holiday"]),
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


def render() -> None:
    """Draw the page.

    The data is kept in this function's own variables, not at the top level
    of the script: those would stay in the script's namespace, which the
    functions defined above keep in memory after the run has ended.
    """
    inject_css()

    st.header("Trends")

    # The data app.py lends each run (see app.main)
    run_data = st.session_state.get("run_data", {})
    if "df_filtered" not in run_data:
        st.warning("Loading data — please wait…")
        st.stop()

    df_filtered: pd.DataFrame = run_data["df_filtered"]

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
    df_all_years: pd.DataFrame = run_data["df_all_years"]

    # ── Daily Incidents (full width) ─────────────────────────────────────────
    st.subheader("Daily Incidents")
    daily = daily_incident_counts(df_filtered, period_start, period_end)

    fig_daily = px.line(daily, x="date", y="count", labels={"date": "Date", "count": "Incidents"})
    fig_daily.update_layout(height=CHART_HEIGHT)
    apply_dark_plotly(fig_daily, BLUE_ACCENT)
    st.plotly_chart(fig_daily, width="stretch")

    # ── Year over Year + Heatmap (side-by-side) ──────────────────────────────
    left, right = st.columns(2)

    with left:
        st.subheader("Year over Year")

        fig_yoy = create_yoy(df_filtered, data_start_date, data_end_date)
        apply_dark_plotly(fig_yoy, BLUE_ACCENT)
        st.plotly_chart(fig_yoy, width="stretch")

        # Which part of a year each lighter bar stands for
        years_shown = set(df_filtered["incident_year"].unique())
        part_years = partial_years(data_start_date, data_end_date)
        part_notes = [
            f"{year} ({part_year_note(*part_years[year])})" for year in sorted(part_years) if year in years_shown
        ]
        if part_notes:
            st.caption(f"Lighter bars cover part of a year: {', '.join(part_notes)}.")

    with right:
        st.subheader("Hour of Day × Day of Week")

        fig_hm, left_out = create_heatmap(df_filtered)
        apply_dark_plotly(fig_hm, BLUE_ACCENT)
        st.plotly_chart(fig_hm, width="stretch")
        # The cells would otherwise add up to less than the other charts, unexplained
        if left_out:
            st.caption(
                f"{left_out:,} of {len(df_filtered):,} incidents have no usable day or hour "
                "and are not in the chart."
            )

    # ── Forecast (full width) ────────────────────────────────────────────────
    st.subheader("1-Year Forecast")

    if selected_year != "All":
        st.caption(
            "The forecast is fitted on every year of data for this selection and starts "
            "after the last day of data, so the Year filter does not apply to it."
        )

    try:
        forecast_fig = create_forecast(df_all_years, data_start_date, data_end_date)
    except Exception as e:
        # The page shows the message only: where it went wrong goes to the server log
        logger.exception("Creating the forecast failed")
        st.error(f"Error creating forecast: {str(e)}")
    else:
        if forecast_fig is None:
            st.info(
                f"Not enough data for forecasting: this selection needs incidents on at least "
                f"{MIN_FORECAST_DAYS} days."
            )
        else:
            apply_dark_plotly(forecast_fig, BLUE_ACCENT)
            st.plotly_chart(forecast_fig, width="stretch")


render()
