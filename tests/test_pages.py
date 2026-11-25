"""Page-level tests, run headlessly with Streamlit's AppTest."""

import base64
import json
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import requests
import streamlit as st
from streamlit.testing.v1 import AppTest

import utils
from conftest import make_page_state
from constants import DAY_LABELS, MAX_SCATTER_POINTS

ROOT = Path(__file__).resolve().parent.parent
TIMEOUT = 300  # the Trends page fits a Prophet model


def run_page(name: str, state: dict) -> AppTest:
    at = AppTest.from_file(str(ROOT / "pages" / name), default_timeout=TIMEOUT)
    for key, value in state.items():
        at.session_state[key] = value
    at.run()
    assert not at.exception, at.exception
    return at


def run_app(feed: pd.DataFrame, monkeypatch) -> AppTest:
    """Run app.py with the download replaced by the given raw feed."""
    monkeypatch.setattr(utils, "safe_load_data_from_url", lambda url, **kwargs: feed.copy())
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=TIMEOUT)
    at.run()
    assert not at.exception, at.exception
    return at


def plotly_figures(at: AppTest) -> list[dict]:
    return [json.loads(chart.proto.spec) for chart in at.get("plotly_chart")]


def values(encoded) -> list:
    """A Plotly array as a flat list, whether sent as plain JSON or packed binary."""
    if isinstance(encoded, dict):
        return np.frombuffer(base64.b64decode(encoded["bdata"]), dtype=encoded["dtype"]).tolist()
    return np.asarray(encoded).ravel().tolist()


def trace(figure: dict, name: str) -> dict:
    return next(t for t in figure["data"] if t.get("name") == name)


def forecast_figure(at: AppTest) -> dict:
    return next(f for f in plotly_figures(at) if any(t.get("name") == "Forecast" for t in f["data"]))


def deck_spec(at: AppTest) -> dict:
    return json.loads(at.get("deck_gl_json_chart")[0].proto.json)


@pytest.fixture(autouse=True)
def clear_caches():
    st.cache_data.clear()
    st.cache_resource.clear()
    yield


# ── Overview (through app.py, with the download replaced) ────────────────────

@pytest.fixture
def overview(raw_feed, monkeypatch) -> AppTest:
    return run_app(raw_feed, monkeypatch)


def test_overview_kpis(overview, clean_df):
    metrics = {m.label: m for m in overview.metric}
    assert metrics["Total Incidents"].value == f"{len(clean_df):,}"
    # Same rate every day in the synthetic feed, so year over year is about flat
    assert metrics["Total Incidents"].delta == metrics["Avg Daily"].delta
    assert abs(float(metrics["Avg Daily"].delta.rstrip("%"))) < 5
    # The date span is shown as plain text, not as a rising delta
    assert metrics["Date Range"].delta == f"{clean_df['date'].min()} to {clean_df['date'].max()}"


def test_overview_footer_reports_data_coverage(overview, clean_df):
    footer = [m.value for m in overview.markdown if 'class="footer-text"' in m.value][0]
    assert f"Data through {clean_df['date'].max()}" in footer


def test_year_filter_only_offers_years_in_the_data(overview, clean_df):
    year_select = overview.sidebar.selectbox[1]
    assert year_select.options == ["All", "2024", "2025", "2026"]

    year_select.select(2025).run()
    assert not overview.exception
    filtered = overview.session_state["df_filtered"]
    assert set(filtered["incident_year"]) == {2025}
    # Every incident's local date falls inside the selected year
    assert filtered["date"].min().year == filtered["date"].max().year == 2025

    # The period is the calendar year, whatever days happen to have incidents
    date_range = {m.label: m for m in overview.metric}["Date Range"]
    assert (date_range.value, date_range.delta) == ("365 days", "2025-01-01 to 2025-12-31")


def test_top_descriptions_are_not_split_by_padding(overview):
    overview.sidebar.selectbox[0].select("CATEGORY 01").run()
    assert not overview.exception
    top_chart = plotly_figures(overview)[0]
    labels = top_chart["data"][0]["y"]
    assert labels == ["CATEGORY 01 - DETAIL"]


def test_rare_category_is_measured_over_the_whole_period(raw_feed, clean_df, monkeypatch):
    # Three incidents on two consecutive days, in a feed that covers 22 months
    feed = raw_feed.copy()
    rare = feed.index[feed["incident_occurred_at"].str.startswith("2025/07/1")][:3]
    feed.loc[rare, "offense_category"] = "RARE"
    feed.loc[rare, "incident_occurred_at"] = [
        "2025/07/10 15:00:00+00", "2025/07/10 18:00:00+00", "2025/07/11 15:00:00+00",
    ]

    at = run_app(feed, monkeypatch)
    at.sidebar.selectbox[0].select("RARE").run()
    assert not at.exception

    first_day, last_day = clean_df["date"].min(), clean_df["date"].max()
    days = (last_day - first_day).days + 1
    metrics = {m.label: m for m in at.metric}
    assert metrics["Total Incidents"].value == "3"
    assert (metrics["Date Range"].value, metrics["Date Range"].delta) == (f"{days:,} days", f"{first_day} to {last_day}")
    assert metrics["Avg Daily"].value == "0.0045"  # 3 incidents in 668 days, not 1.5 a day

    # The daily chart shows the quiet days too
    daily = plotly_figures(at)[-1]["data"][0]
    assert len(daily["x"]) == days
    assert sum(values(daily["y"])) == 3


# ── Loading failures ─────────────────────────────────────────────────────────

def portal_down(url, **kwargs):
    raise requests.exceptions.ConnectionError("portal down")


def test_first_load_failure_is_reported(monkeypatch):
    monkeypatch.setattr(utils, "safe_load_data_from_url", portal_down)
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=TIMEOUT)
    at.run()
    assert [e.value for e in at.error] == [
        "Connection failed. Please check your internet connection and try again."
    ]
    assert not at.metric


def test_failed_refresh_keeps_showing_the_last_data(overview, monkeypatch):
    # The data is due for a refresh, and the portal is down
    monkeypatch.setattr(utils, "safe_load_data_from_url", portal_down)
    utils.dataset_cache().expire()
    overview.run()

    assert not overview.exception and not overview.error
    assert [w.value for w in overview.warning] == [
        "The data could not be refreshed, so the last copy that loaded is shown. "
        "Connection failed. Please check your internet connection and try again."
    ]
    assert {m.label for m in overview.metric} >= {"Total Incidents", "Avg Daily"}


def test_feed_with_no_usable_rows_is_an_error_and_is_not_kept(raw_feed, monkeypatch):
    # Every coordinate outside Detroit: the feed passes validation, then loses every row
    monkeypatch.setattr(utils, "safe_load_data_from_url", lambda url, **kwargs: raw_feed.assign(latitude=10.0))
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=TIMEOUT)
    at.run()
    assert [e.value for e in at.error] == [
        "Data validation error: no incidents were left after cleaning; "
        "the dates or coordinates in the feed may have changed"
    ]
    assert not at.metric

    # Nothing was cached: once the feed is right again, the next run loads it
    monkeypatch.setattr(utils, "safe_load_data_from_url", lambda url, **kwargs: raw_feed.copy())
    at.run()
    assert not at.error
    assert at.metric


# ── Trends ───────────────────────────────────────────────────────────────────

def test_trends_heatmap_labels_days_iso(page_state):
    at = run_page("2_Trends.py", page_state)
    figures = plotly_figures(at)
    heatmap = next(f for f in figures if f["data"][0]["type"] == "heatmap")
    yaxis = heatmap["layout"]["yaxis"]
    assert yaxis["tickvals"] == [1, 2, 3, 4, 5, 6, 7]
    assert yaxis["ticktext"] == DAY_LABELS
    assert DAY_LABELS[0] == "Monday" and DAY_LABELS[6] == "Sunday"

    # Daily, year over year, heatmap and the forecast all rendered
    assert len(figures) == 4
    assert not at.error
    assert not at.caption  # the note about the Year filter is only shown with one


def test_heatmap_always_shows_the_full_week_and_day(clean_df):
    # A dozen incidents cannot cover 7 days x 24 hours; the grid is complete anyway
    few = clean_df[clean_df["offense_category"] == "CATEGORY 20"].head(12)
    at = run_page("2_Trends.py", dict(make_page_state(clean_df), df_filtered=few))
    heatmap = next(f for f in plotly_figures(at) if f["data"][0]["type"] == "heatmap")["data"][0]
    assert values(heatmap["x"]) == list(range(24))
    assert values(heatmap["y"]) == [1, 2, 3, 4, 5, 6, 7]
    assert sum(values(heatmap["z"])) == 12


def test_forecast_ignores_the_year_filter(clean_df):
    at = run_page("2_Trends.py", make_page_state(clean_df, year=2025))
    figures = plotly_figures(at)

    # The daily chart covers the selected year...
    daily = figures[0]["data"][0]
    assert (daily["x"][0][:10], daily["x"][-1][:10]) == ("2025-01-01", "2025-12-31")

    # ...but the forecast starts where the data really ends, not on 1 Jan 2026
    data_end = clean_df["date"].max()
    forecast = trace(forecast_figure(at), "Forecast")
    assert forecast["x"][0][:10] == str(data_end + timedelta(days=1))
    assert len(forecast["x"]) == 365
    assert "Year filter does not apply" in at.caption[0].value

    # A single year is ticked as a whole number
    yoy = next(f for f in figures if f["data"][0]["type"] == "bar")
    assert values(yoy["data"][0]["x"]) == [2025]
    assert yoy["layout"]["xaxis"]["dtick"] == 1


def test_forecast_for_a_quiet_category_starts_after_the_data_and_stays_above_zero(clean_df):
    # The category's last incident is two months before the data ends
    quiet = "CATEGORY 20"
    late = (clean_df["offense_category"] == quiet) & (clean_df["incident_occurred_at"] >= "2026-02-01")
    df = clean_df[~late]
    at = run_page("2_Trends.py", make_page_state(df, category=quiet))
    figure = forecast_figure(at)

    data_end = df["date"].max()
    assert df.loc[df["offense_category"] == quiet, "date"].max() < data_end - timedelta(days=50)
    assert trace(figure, "Forecast")["x"][0][:10] == str(data_end + timedelta(days=1))

    # About one incident every three weeks: an unclipped band would dip below zero
    for band in figure["data"][1:]:
        assert min(values(band["y"])) >= 0


def test_trends_projection_starts_from_last_day_of_data(page_state, clean_df):
    at = run_page("2_Trends.py", page_state)
    yoy = next(f for f in plotly_figures(at) if f["data"][0]["type"] == "bar")
    actual, projected = yoy["data"]
    assert projected["name"] == "Projected"
    assert yoy["layout"]["xaxis"]["dtick"] == 1

    as_of = page_state["data_end_date"]
    current = int((clean_df["incident_year"] == as_of.year).sum())
    expected_total = utils.project_full_year(current, as_of)
    assert projected["base"][0] + projected["y"][0] == pytest.approx(expected_total)


def test_forecast_failure_is_reported_and_not_cached(page_state, monkeypatch):
    from prophet import Prophet

    def broken_fit(self, *args, **kwargs):
        raise RuntimeError("optimizer exploded")

    with monkeypatch.context() as patched:
        patched.setattr(Prophet, "fit", broken_fit)
        at = run_page("2_Trends.py", page_state)
        assert [e.value for e in at.error] == ["Error creating forecast: optimizer exploded"]
        assert not at.info  # not blamed on "not enough data"

    # Next visit, with Prophet working again, gets a real forecast
    at = run_page("2_Trends.py", page_state)
    assert not at.error
    assert len(plotly_figures(at)) == 4


# ── Pivot analysis ───────────────────────────────────────────────────────────

def test_swap_button_swaps_rows_and_columns(page_state):
    at = run_page("4_Analysis.py", page_state)
    rows, columns = at.selectbox(key="pivot_rows"), at.selectbox(key="pivot_columns")
    assert (rows.value, columns.value) == ("offense_category", "case_status")
    before = at.dataframe[0].value

    at.button(key="swap_pivot").click().run()
    assert not at.exception
    assert at.selectbox(key="pivot_rows").value == "case_status"
    assert at.selectbox(key="pivot_columns").value == "offense_category"
    # The table itself is transposed on the same click
    after = at.dataframe[0].value
    assert after.shape == before.shape[::-1]
    assert after.index.name == "case_status"

    at.button(key="swap_pivot").click().run()
    assert at.selectbox(key="pivot_rows").value == "offense_category"


def test_pivot_selection_is_kept_for_the_next_visit(page_state):
    at = run_page("4_Analysis.py", page_state)
    at.selectbox(key="pivot_rows").select("neighborhood").run()
    assert at.session_state["pivot_rows_selection"] == "neighborhood"

    # Coming back to the page: widget state is gone, the saved selection is not
    state = dict(page_state, pivot_rows_selection="neighborhood", pivot_columns_selection="zip_code")
    at = run_page("4_Analysis.py", state)
    assert at.selectbox(key="pivot_rows").value == "neighborhood"
    assert at.selectbox(key="pivot_columns").value == "zip_code"


def test_pivot_sorts_codes_as_numbers(page_state):
    at = run_page("4_Analysis.py", page_state)
    at.selectbox(key="pivot_rows").select("police_precinct").run()
    at.selectbox(key="pivot_columns").select("zip_code").run()
    assert not at.exception and not at.error

    table = at.dataframe[0].value
    # Precinct 2 comes before Precinct 10; as text it would come after Precinct 12
    assert table.index.tolist() == [f"Precinct {n}" for n in range(2, 13)]
    assert table.columns.tolist() == ["48201", "48207", "48210", "48226"]

    # Zip codes are category labels, not points on a number line
    figure = plotly_figures(at)[0]
    assert figure["layout"]["xaxis"]["type"] == "category"


def test_pivot_counts_every_incident(page_state, clean_df):
    # Rows are counted, so an incident without an entry id is still included
    no_ids = dict(page_state, df_filtered=clean_df.assign(incident_entry_id=None))
    at = run_page("4_Analysis.py", no_ids)
    assert not at.error
    assert int(at.dataframe[0].value.to_numpy().sum()) == len(clean_df)


def test_pivot_year_axis_ticks_on_whole_years(clean_df):
    at = run_page("4_Analysis.py", make_page_state(clean_df, year=2025))
    at.selectbox(key="pivot_columns").select("incident_year").run()
    assert not at.error
    figure = plotly_figures(at)[0]
    assert figure["layout"]["xaxis"]["dtick"] == 1
    assert at.dataframe[0].value.columns.tolist() == [2025]


# ── Geography ────────────────────────────────────────────────────────────────

def test_geography_charts(page_state):
    at = run_page("3_Geography.py", page_state)
    figures = plotly_figures(at)
    assert len(figures) == 5
    precincts = figures[1]["data"][0]["y"]
    assert all(label.startswith("Precinct ") for label in precincts)
    zips = figures[3]["data"][0]["y"]
    assert set(zips) == {"48201", "48226", "48207", "48210"}


# ── Map ──────────────────────────────────────────────────────────────────────

def legend_colors(at: AppTest) -> dict:
    """Legend label -> css color, parsed from the legend markup."""
    import re

    legend = [m.value for m in at.markdown if "border-radius:50%" in m.value][0]
    pairs = re.findall(r'background:(rgb\([^)]*\)).*?<span style="margin-right:14px;">(.*?)</span>', legend)
    return {label: color for color, label in pairs}


def test_map_colors_are_unique_and_stable(page_state, clean_df):
    everything = legend_colors(run_page("5_Map.py", page_state))

    # 20 categories, 15 palette colors: 15 named entries plus one shared "Other"
    assert "Other (5 categories)" in everything
    assert len(everything) == 16
    assert len(set(everything.values())) == 16

    # Filtered to one category, it keeps the color it has in the full view
    category = page_state["category_order"][3]
    only = dict(page_state, df_filtered=clean_df[clean_df["offense_category"] == category])
    assert legend_colors(run_page("5_Map.py", only)) == {category: everything[category]}


def test_map_scatter_payload(page_state, clean_df):
    at = run_page("5_Map.py", page_state)
    raw_json = at.get("deck_gl_json_chart")[0].proto.json
    assert "NaN" not in raw_json

    points = deck_spec(at)["layers"][0]["data"]
    assert len(points) == len(clean_df)
    assert set(points[0]) == {"lat", "lon", "category", "location", "r", "g", "b"}
    assert at.caption[0].value == f"{len(clean_df):,} geocoded incidents"

    tooltip = json.loads(at.get("deck_gl_json_chart")[0].proto.tooltip)
    assert tooltip["text"] == "{category}\n{location}"
    assert "html" not in tooltip


def test_map_samples_very_wide_selections(page_state, clean_df):
    too_many = pd.concat([clean_df] * (MAX_SCATTER_POINTS // len(clean_df) + 1), ignore_index=True)
    at = run_page("5_Map.py", dict(page_state, df_filtered=too_many))

    assert len(deck_spec(at)["layers"][0]["data"]) == MAX_SCATTER_POINTS
    assert at.caption[0].value.startswith(f"Showing a random sample of {MAX_SCATTER_POINTS:,} of {len(too_many):,}")


def test_map_heatmap_weights_cover_every_incident(page_state, clean_df):
    at = run_page("5_Map.py", page_state)
    at.radio[0].set_value("Heatmap").run()
    assert not at.exception

    layer = deck_spec(at)["layers"][0]
    assert layer["@@type"] == "HeatmapLayer"
    assert sum(point["weight"] for point in layer["data"]) == len(clean_df)
