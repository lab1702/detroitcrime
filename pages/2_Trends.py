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
