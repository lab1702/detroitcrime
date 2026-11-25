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
from utils import project_full_year
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

# Last day the dataset covers; "today" for year-to-date projections
data_end_date = st.session_state["data_end_date"]


# ── Daily Incidents (full width) ─────────────────────────────────────────────
st.subheader("Daily Incidents")
daily = df_filtered.groupby("date").size().reset_index(name="count")
daily["date"] = pd.to_datetime(daily["date"])
date_range = pd.date_range(start=daily["date"].min(), end=daily["date"].max(), freq="D")
daily = pd.DataFrame({"date": date_range}).merge(daily, on="date", how="left").fillna(0)

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
        ).fillna(0)

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
def create_forecast(df: pd.DataFrame) -> Optional[go.Figure]:
    """Create Prophet forecast for incident count.

    Returns None when there is too little history to forecast. Errors are
    raised, not swallowed, so a failed fit is never cached as a result.
    """
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


try:
    forecast_fig = create_forecast(df_filtered)
except Exception as e:
    st.error(f"Error creating forecast: {str(e)}")
else:
    if forecast_fig is None:
        st.info(f"Not enough data for forecasting (minimum {MIN_FORECAST_DAYS} days required).")
    else:
        apply_dark_plotly(forecast_fig, BLUE_ACCENT)
        st.plotly_chart(forecast_fig, width="stretch")
