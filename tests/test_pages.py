"""Page-level tests, run headlessly with Streamlit's AppTest."""

import json
from pathlib import Path

import pandas as pd
import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

import utils
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


def plotly_figures(at: AppTest) -> list[dict]:
    return [json.loads(chart.proto.spec) for chart in at.get("plotly_chart")]


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
    monkeypatch.setattr(utils, "safe_load_data_from_url", lambda url, **kwargs: raw_feed.copy())
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=TIMEOUT)
    at.run()
    assert not at.exception, at.exception
    return at


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


def test_top_descriptions_are_not_split_by_padding(overview):
    overview.sidebar.selectbox[0].select("CATEGORY 01").run()
    assert not overview.exception
    top_chart = plotly_figures(overview)[0]
    labels = top_chart["data"][0]["y"]
    assert labels == ["CATEGORY 01 - DETAIL"]


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


def test_trends_projection_starts_from_last_day_of_data(page_state, clean_df):
    at = run_page("2_Trends.py", page_state)
    yoy = next(f for f in plotly_figures(at) if f["data"][0]["type"] == "bar")
    actual, projected = yoy["data"]
    assert projected["name"] == "Projected"

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
