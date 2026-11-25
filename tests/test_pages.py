"""Page-level tests, run headlessly with Streamlit's AppTest."""

import base64
import json
import threading
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import requests
import streamlit as st
from streamlit.proto.Metric_pb2 import Metric as MetricProto
from streamlit.testing.v1 import AppTest

import utils
from conftest import clean_feed, make_page_state, make_raw_feed
from constants import DAY_LABELS, MAX_SCATTER_POINTS

ROOT = Path(__file__).resolve().parent.parent
TIMEOUT = 300  # the Trends page fits a Prophet model

# What a page says while it waits for a download to end
REFRESHING_NOTE = "Newer data is being downloaded. This page will update when it is ready."
RETRY_NOTE = "The download will be tried again shortly, and this page will update by itself."


def run_page(name: str, state: dict) -> AppTest:
    at = AppTest.from_file(str(ROOT / "views" / name), default_timeout=TIMEOUT)
    # As app.py hands it over: the DataFrames in the holder it lends each run,
    # the rest as plain session state
    run_data = {key: value for key, value in state.items() if isinstance(value, pd.DataFrame)}
    at.session_state["run_data"] = run_data
    for key, value in state.items():
        if key not in run_data:
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


def held_back(result):
    """A download that does not end until it is let go.

    Returns the stand-in for the download and the Event that releases it.
    `result` is the raw feed it then returns, or the exception it raises.
    """
    release = threading.Event()

    def download(url, **kwargs):
        release.wait(timeout=TIMEOUT)
        if isinstance(result, Exception):
            raise result
        return result.copy()

    return download, release


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


# ── Navigation ───────────────────────────────────────────────────────────────

def test_page_scripts_are_not_kept_in_a_pages_folder():
    # Until st.navigation has been called once, Streamlit runs the scripts of a
    # pages/ folder on their own, without app.py (see the note in app.main)
    assert not (ROOT / "pages").exists()


def test_first_visit_on_a_page_url_goes_through_the_app(raw_feed, monkeypatch):
    # A new server whose first visitor arrives on /Trends, not on the front page
    monkeypatch.setattr(utils, "safe_load_data_from_url", lambda url, **kwargs: raw_feed.copy())
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=TIMEOUT)
    at.switch_page("views/2_Trends.py").run()

    assert not at.exception, at.exception
    assert "Trends" in [h.value for h in at.header]
    # app.py ran first: the data is loaded, the filters are there, the charts are drawn
    assert not at.warning
    assert [s.label for s in at.sidebar.selectbox] == ["Offense Category", "Year"]
    assert len(plotly_figures(at)) == 4


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
    metrics = {m.label: m for m in overview.metric}
    # Only the incidents whose local date falls inside the selected year are counted
    in_2025 = clean_df["date"].map(lambda day: day.year == 2025)
    assert metrics["Total Incidents"].value == f"{in_2025.sum():,}"

    # The period is the calendar year, whatever days happen to have incidents
    date_range = metrics["Date Range"]
    assert (date_range.value, date_range.delta) == ("365 days", "2025-01-01 to 2025-12-31")


def test_help_explains_the_change_only_where_one_is_shown(overview):
    explains_change = "The change compares this year"
    for_every_year = {m.label: m.help for m in overview.metric}
    assert explains_change in for_every_year["Total Incidents"]
    assert explains_change in for_every_year["Avg Daily"]

    # A past year has no change from last year, so there is none to explain
    overview.sidebar.selectbox[1].select(2025).run()
    past_year = {m.label: m for m in overview.metric}
    assert not past_year["Total Incidents"].delta
    assert explains_change not in past_year["Total Incidents"].help
    assert not past_year["Avg Daily"].help
    # What the total leaves out is said whatever the year
    assert "coordinates" in past_year["Total Incidents"].help
    assert "coordinates" in for_every_year["Total Incidents"]


def test_no_dataframe_is_left_in_session_state(raw_feed, monkeypatch):
    # Session state lasts as long as the browser tab stays open. The data is
    # lent to the page for one run, not kept there for every idle tab.
    feed = raw_feed.copy()
    rare = feed.index[feed["incident_occurred_at"].str.startswith("2025/07/1")][:3]
    feed.loc[rare, "offense_category"] = "RARE"
    at = run_app(feed, monkeypatch)
    assert at.metric  # the page had its data while it ran
    assert at.session_state["run_data"] == {}
    assert "df_filtered" not in at.session_state and "df_all_years" not in at.session_state

    # The same after a run the page cut short: this category has nothing in 2026
    at.sidebar.selectbox[0].select("RARE").run()
    at.sidebar.selectbox[1].select(2026).run()
    assert [w.value for w in at.warning] == ["No data available for selected filters."]
    assert at.session_state["run_data"] == {}

    # And after a run on another page
    at.switch_page("views/3_Geography.py").run()
    assert not at.exception
    assert at.session_state["run_data"] == {}


def test_current_year_keeps_its_change_from_last_year(overview):
    for_every_year = {m.label: m.delta for m in overview.metric}
    assert for_every_year["Avg Daily"].endswith("%")

    # The Year filter leaves only 2026 in the selection, but not in the comparison
    overview.sidebar.selectbox[1].select(2026).run()
    assert not overview.exception
    this_year = {m.label: m.delta for m in overview.metric}
    assert this_year["Total Incidents"] == for_every_year["Total Incidents"]
    assert this_year["Avg Daily"] == for_every_year["Avg Daily"]

    # A past year is not what the comparison is about, so none is claimed for it
    overview.sidebar.selectbox[1].select(2025).run()
    past_year = {m.label: m.delta for m in overview.metric}
    assert not past_year["Total Incidents"] and not past_year["Avg Daily"]


def test_no_change_from_last_year_unless_the_data_covers_it_from_its_first_day(monkeypatch):
    # The same 8 incidents a day throughout, in a feed that starts on 15
    # February last year. Its 44 days up to 30 March, set against this year's
    # 89, would show as crime having doubled.
    at = run_app(make_raw_feed(start="2025-02-15"), monkeypatch)
    metrics = {m.label: m for m in at.metric}
    assert not metrics["Total Incidents"].delta and not metrics["Avg Daily"].delta
    assert "The change compares this year" not in metrics["Total Incidents"].help


def test_change_that_rounds_to_nothing_is_not_drawn_as_a_rise(overview):
    # The same number of incidents as in the same days of last year
    for label in ("Total Incidents", "Avg Daily"):
        metric = {m.label: m for m in overview.metric}[label]
        assert metric.delta == "0.0%"
        assert metric.proto.direction == MetricProto.MetricDirection.NONE
        assert metric.proto.color == MetricProto.MetricColor.GRAY


def test_rise_from_last_year_is_drawn_as_bad_news(raw_feed, monkeypatch):
    # Ten days fewer of last year's incidents: this year's count is the higher
    fewer_last_year = raw_feed[~raw_feed["incident_occurred_at"].str.startswith("2025/02/1")]
    at = run_app(fewer_last_year, monkeypatch)
    for label in ("Total Incidents", "Avg Daily"):
        metric = {m.label: m for m in at.metric}[label]
        assert metric.delta.startswith("+")
        assert metric.proto.direction == MetricProto.MetricDirection.UP
        assert metric.proto.color == MetricProto.MetricColor.RED


def test_top_descriptions_are_not_split_by_padding(overview):
    overview.sidebar.selectbox[0].select("CATEGORY 01").run()
    assert not overview.exception
    top_chart = plotly_figures(overview)[0]
    labels = top_chart["data"][0]["y"]
    assert labels == ["CATEGORY 01 - DETAIL"]


def test_overview_without_any_category(raw_feed, clean_df, monkeypatch):
    # The category column arrives empty: there is no most common one to name,
    # and nothing for the categories chart to draw
    at = run_app(raw_feed.assign(offense_category=np.nan), monkeypatch)
    metrics = {m.label: m.value for m in at.metric}
    assert metrics["Most Common"] == "—"
    assert metrics["Total Incidents"] == f"{len(clean_df):,}"
    assert [i.value for i in at.info] == ["Nothing recorded for this selection."]
    # Case status and daily incidents are drawn as usual
    assert len(plotly_figures(at)) == 2


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
    # A fault on the line can pass: the page says it will look again by itself
    assert [c.value for c in at.caption] == [RETRY_NOTE]


def test_passing_first_load_failure_is_tried_again_without_a_reload(raw_feed, monkeypatch):
    # No wait before another try, so the page is at once due to look again
    import constants

    monkeypatch.setattr(constants, "DATA_FIRST_LOAD_RETRY_INTERVAL", 0)
    monkeypatch.setattr(utils, "DATA_FIRST_LOAD_RETRY_INTERVAL", 0)
    downloads = []

    def down_then_up(url, **kwargs):
        downloads.append(url)
        if len(downloads) == 1:
            raise requests.exceptions.ConnectionError("portal down")
        return raw_feed.copy()

    monkeypatch.setattr(utils, "safe_load_data_from_url", down_then_up)
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=TIMEOUT)
    at.run()

    # One run from the visitor: the page went back for the data by itself
    assert not at.exception, at.exception
    assert len(downloads) == 2
    assert not at.error and not at.caption
    assert at.metric


def test_dataset_that_has_moved_is_not_reported_as_a_passing_fault(monkeypatch):
    def not_found(url, **kwargs):
        response = requests.Response()
        response.status_code = 404
        raise requests.exceptions.HTTPError("404 Client Error: Not Found for url: https://example.invalid/feed.csv", response=response)

    monkeypatch.setattr(utils, "safe_load_data_from_url", not_found)
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=TIMEOUT)
    at.run()
    assert [e.value for e in at.error] == [
        "The data portal would not serve the dataset: 404 Client Error: Not Found for url: "
        "https://example.invalid/feed.csv. It may have been moved or withdrawn, in which case "
        "the address the app downloads from needs updating."
    ]
    # It would say the same on every try, so the page does not go on asking
    assert not at.caption


def test_server_fault_is_reported_as_one_that_may_pass(monkeypatch):
    def unavailable(url, **kwargs):
        response = requests.Response()
        response.status_code = 503
        raise requests.exceptions.HTTPError("503 Server Error: Service Unavailable", response=response)

    monkeypatch.setattr(utils, "safe_load_data_from_url", unavailable)
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=TIMEOUT)
    at.run()
    assert [e.value for e in at.error] == [
        "Server error: 503 Server Error: Service Unavailable. The data source may be temporarily unavailable."
    ]
    assert [c.value for c in at.caption] == [RETRY_NOTE]


def test_failed_first_load_is_not_repeated_by_the_next_rerun(monkeypatch):
    downloads = []

    def portal_down_counted(url, **kwargs):
        downloads.append(url)
        raise requests.exceptions.ConnectionError("portal down")

    monkeypatch.setattr(utils, "safe_load_data_from_url", portal_down_counted)
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=TIMEOUT)
    at.run()
    at.run()

    # The second run reports the same failure without downloading again
    assert len(downloads) == 1
    assert [e.value for e in at.error] == [
        "Connection failed. Please check your internet connection and try again."
    ]


def test_download_cut_short_is_reported_in_plain_words(monkeypatch):
    def cut_short(url, **kwargs):
        raise requests.exceptions.ChunkedEncodingError("Connection broken: IncompleteRead(1024 bytes read)")

    monkeypatch.setattr(utils, "safe_load_data_from_url", cut_short)
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=TIMEOUT)
    at.run()
    assert [e.value for e in at.error] == ["The download was interrupted before it finished. Please try again."]


def test_failed_refresh_keeps_showing_the_last_data(overview, monkeypatch):
    # The data is due for a refresh, and the portal is down (it takes a while to say so)
    download, release = held_back(requests.exceptions.ConnectionError("portal down"))
    monkeypatch.setattr(utils, "safe_load_data_from_url", download)
    utils.dataset_cache().expire()

    # The refresh runs behind the page, which shows the data it has without
    # waiting, and says that newer data is on its way
    overview.run()
    assert not overview.exception and not overview.error and not overview.warning
    assert [c.value for c in overview.caption] == [REFRESHING_NOTE]
    assert {m.label for m in overview.metric} >= {"Total Incidents", "Avg Daily"}

    # Once the refresh has failed the next run says so, and still shows the data
    release.set()
    assert utils.dataset_cache().wait(timeout=TIMEOUT)
    overview.run()
    assert not overview.exception and not overview.error
    assert [w.value for w in overview.warning] == [
        "The data could not be refreshed, so the last copy that loaded is shown. "
        "Connection failed. Please check your internet connection and try again."
    ]
    assert not overview.caption  # nothing is being downloaded any more
    assert {m.label for m in overview.metric} >= {"Total Incidents", "Avg Daily"}


def test_refreshed_data_is_shown_on_the_run_after_it_arrives(overview, raw_feed, monkeypatch):
    # The feed has lost its last month by the time the data is due for a refresh
    shorter = raw_feed[raw_feed["incident_occurred_at"] < "2026/03/01"]
    download, release = held_back(shorter)
    monkeypatch.setattr(utils, "safe_load_data_from_url", download)
    before = {m.label: m.value for m in overview.metric}["Total Incidents"]
    utils.dataset_cache().expire()

    overview.run()
    assert {m.label: m.value for m in overview.metric}["Total Incidents"] == before
    assert [c.value for c in overview.caption] == [REFRESHING_NOTE]

    release.set()
    assert utils.dataset_cache().wait(timeout=TIMEOUT)
    overview.run()
    assert not overview.exception and not overview.warning and not overview.caption
    assert {m.label: m.value for m in overview.metric}["Total Incidents"] != before


class WaitingCache:
    """Stands in for the dataset cache while a refresh is said to be running.

    `still_running` is what it answers, in turn, each time it is asked whether
    the refresh is still under way; after that the refresh has ended.
    """

    def __init__(self, df: pd.DataFrame, still_running: list[bool]):
        self.df = df
        self.still_running = still_running
        self.gets = 0

    def get(self, loader):
        self.gets += 1
        return self.df, None

    def wait(self, timeout=None) -> bool:
        return not (self.still_running.pop(0) if self.still_running else False)


def test_page_is_drawn_again_when_the_refresh_it_waits_for_has_ended(clean_df, monkeypatch):
    # Running when the page starts, over by the time the page asks again. An
    # open page asks every few seconds; here it is on its first time of asking.
    cache = WaitingCache(clean_df, still_running=[True, False])
    monkeypatch.setattr(utils, "dataset_cache", lambda: cache)
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=TIMEOUT)
    at.run()

    assert not at.exception, at.exception
    # The page went back for the data and drew itself again, with nothing left to wait for
    assert cache.gets == 2
    assert not at.caption
    assert at.metric


def test_page_waiting_for_a_refresh_is_not_drawn_again_before_it_ends(clean_df, monkeypatch):
    cache = WaitingCache(clean_df, still_running=[True, True])
    monkeypatch.setattr(utils, "dataset_cache", lambda: cache)
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=TIMEOUT)
    at.run()

    assert not at.exception, at.exception
    assert cache.gets == 1
    assert [c.value for c in at.caption] == [REFRESHING_NOTE]
    assert at.metric


def test_refresh_that_has_lost_most_of_its_incidents_is_refused(overview, raw_feed, monkeypatch):
    before = {m.label: m.value for m in overview.metric}

    # Nine in ten rows come back without coordinates, as after a geocoding fault
    gutted = raw_feed.copy()
    lost = gutted.sample(frac=0.9, random_state=1).index
    gutted.loc[lost, ["latitude", "longitude"]] = np.nan
    monkeypatch.setattr(utils, "safe_load_data_from_url", lambda url, **kwargs: gutted.copy())
    utils.dataset_cache().expire()
    overview.run()
    assert utils.dataset_cache().wait(timeout=TIMEOUT)
    overview.run()
    assert not overview.exception and not overview.error

    # The copy in use is kept, and the page says why the new one was not taken
    assert {m.label: m.value for m in overview.metric} == before
    [warning] = [w.value for w in overview.warning]
    assert warning.startswith(
        "The data could not be refreshed, so the last copy that loaded is shown. "
        "Data validation error: the refreshed feed has only "
    )
    assert f"against {before['Total Incidents']} in the copy shown" in warning


def test_filters_are_kept_when_a_refresh_changes_their_options(raw_feed, monkeypatch):
    # A feed with only the last month of 2024 (local midnight on 1 Dec is 05:00 UTC)
    first = raw_feed[raw_feed["incident_occurred_at"] >= "2024/12/01 05:00:00+00"]
    overview = run_app(first, monkeypatch)
    assert overview.sidebar.selectbox[1].options == ["All", "2024", "2025", "2026"]
    overview.sidebar.selectbox[0].select("CATEGORY 03").run()
    overview.sidebar.selectbox[1].select(2025).run()
    before = {m.label: m.value for m in overview.metric}["Total Incidents"]

    # The refreshed feed has lost 2024 (local midnight on 1 Jan 2025 is 05:00 UTC)
    # and has a category not seen before, on a day in 2026
    newer = raw_feed[raw_feed["incident_occurred_at"] >= "2025/01/01 05:00:00+00"].copy()
    in_2026 = newer.index[newer["incident_occurred_at"].str.startswith("2026/02")]
    newer.loc[in_2026[0], "offense_category"] = "BRAND NEW"
    monkeypatch.setattr(utils, "safe_load_data_from_url", lambda url, **kwargs: newer.copy())
    utils.dataset_cache().expire()
    overview.run()
    assert utils.dataset_cache().wait(timeout=TIMEOUT)
    overview.run()
    assert not overview.exception

    # Both lists of options changed, and both choices are still made
    category, year = overview.sidebar.selectbox
    assert "BRAND NEW" in category.options
    assert year.options == ["All", "2025", "2026"]
    assert (category.value, year.value) == ("CATEGORY 03", 2025)
    assert {m.label: m.value for m in overview.metric}["Total Incidents"] == before


def test_filter_that_is_no_longer_offered_falls_back_to_all(overview, raw_feed, monkeypatch):
    overview.sidebar.selectbox[0].select("CATEGORY 12").run()
    overview.sidebar.selectbox[1].select(2025).run()

    # The category has gone from the refreshed feed (one in twenty incidents:
    # a feed that had lost many more would be refused as incomplete)
    without = raw_feed[raw_feed["offense_category"] != "CATEGORY 12"]
    monkeypatch.setattr(utils, "safe_load_data_from_url", lambda url, **kwargs: without.copy())
    utils.dataset_cache().expire()
    overview.run()
    assert utils.dataset_cache().wait(timeout=TIMEOUT)
    overview.run()
    assert not overview.exception

    category, year = overview.sidebar.selectbox
    assert "CATEGORY 12" not in category.options
    assert (category.value, year.value) == ("All", 2025)
    in_2025 = clean_feed(without)["incident_year"] == 2025
    assert {m.label: m.value for m in overview.metric}["Total Incidents"] == f"{in_2025.sum():,}"


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
    # The same feed would fail the same way, at the cost of a whole download
    # each time: the page does not try again by itself
    assert not at.caption

    # Nothing was cached: once the feed is right again, and the wait before
    # another try is over, the next run loads it
    monkeypatch.setattr(utils, "safe_load_data_from_url", lambda url, **kwargs: raw_feed.copy())
    utils.dataset_cache().expire()
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
    # The note about the Year filter is only shown with one
    assert not any("Year filter" in c.value for c in at.caption)


def test_heatmap_always_shows_the_full_week_and_day(clean_df):
    # A dozen incidents cannot cover 7 days x 24 hours; the grid is complete anyway
    few = clean_df[clean_df["offense_category"] == "CATEGORY 20"].head(12)
    at = run_page("2_Trends.py", dict(make_page_state(clean_df), df_filtered=few))
    heatmap = next(f for f in plotly_figures(at) if f["data"][0]["type"] == "heatmap")["data"][0]
    assert values(heatmap["x"]) == list(range(24))
    assert values(heatmap["y"]) == [1, 2, 3, 4, 5, 6, 7]
    assert sum(values(heatmap["z"])) == 12


def test_heatmap_counts_the_rest_when_a_day_cell_is_not_a_number(raw_feed):
    # One bad cell makes read_csv hand over the whole day column as text
    feed = raw_feed.assign(incident_day_of_week=raw_feed["incident_day_of_week"].astype(str))
    feed.loc[feed.index[0], "incident_day_of_week"] = "UNK"
    df = clean_feed(feed)

    at = run_page("2_Trends.py", make_page_state(df))
    heatmap = next(f for f in plotly_figures(at) if f["data"][0]["type"] == "heatmap")["data"][0]
    assert values(heatmap["y"]) == [1, 2, 3, 4, 5, 6, 7]
    assert sum(values(heatmap["z"])) == len(df) - 1


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
    quiet = "CATEGORY 19"
    late = (clean_df["offense_category"] == quiet) & (clean_df["incident_occurred_at"] >= "2026-02-01")
    df = clean_df[~late]
    at = run_page("2_Trends.py", make_page_state(df, category=quiet))
    figure = forecast_figure(at)

    data_end = df["date"].max()
    assert df.loc[df["offense_category"] == quiet, "date"].max() < data_end - timedelta(days=50)
    assert trace(figure, "Forecast")["x"][0][:10] == str(data_end + timedelta(days=1))

    # About one incident every two weeks: an unclipped band would dip below zero
    for band in figure["data"][1:]:
        assert min(values(band["y"])) >= 0


def test_no_forecast_for_a_selection_with_too_little_history(clean_df):
    # Three incidents in 22 months. There is a day in the series for every day
    # the data covers, but only three of them say anything about this category.
    df = clean_df.copy()
    rare = df.index[df["incident_occurred_at"] >= "2025-07-10"][:3]
    df.loc[rare, "offense_category"] = "RARE"

    at = run_page("2_Trends.py", make_page_state(df, category="RARE"))
    assert not at.error
    assert [i.value for i in at.info] == [
        "Not enough data for forecasting: this selection needs incidents on at least 30 days."
    ]
    # Daily, year over year and the heatmap are drawn; the forecast is not
    figures = plotly_figures(at)
    assert len(figures) == 3
    assert not any(t.get("name") == "Forecast" for f in figures for t in f["data"])


def test_trends_projection_starts_from_last_day_of_data(page_state, clean_df):
    at = run_page("2_Trends.py", page_state)
    yoy = next(f for f in plotly_figures(at) if f["data"][0]["type"] == "bar")
    actual, projected = yoy["data"]
    assert projected["name"] == "Projected"
    assert yoy["layout"]["xaxis"]["dtick"] == 1

    as_of = page_state["data_end_date"]
    current = int((clean_df["incident_year"] == as_of.year).sum())
    expected_total = utils.project_full_year(current, as_of, page_state["data_start_date"])
    assert projected["base"][0] + projected["y"][0] == pytest.approx(expected_total)


def test_part_years_are_marked_on_the_year_over_year_chart(page_state):
    # The data runs from 1 June 2024 to 30 March 2026: only 2025 is a whole year
    at = run_page("2_Trends.py", page_state)
    yoy = next(f for f in plotly_figures(at) if f["data"][0]["type"] == "bar")
    bars = yoy["data"][0]
    assert values(bars["x"]) == [2024, 2025, 2026]
    assert values(bars["marker"]["opacity"]) == [0.45, 1, 0.45]
    assert [c.value for c in at.caption] == [
        "Lighter bars cover part of a year: 2024 (from 1 Jun), 2026 (to 30 Mar)."
    ]

    # A whole year on its own is neither lighter nor explained
    at = run_page("2_Trends.py", make_page_state(page_state["df_all_years"], year=2025))
    yoy = next(f for f in plotly_figures(at) if f["data"][0]["type"] == "bar")
    assert values(yoy["data"][0]["marker"]["opacity"]) == [1]
    assert not any("Lighter bars" in c.value for c in at.caption)


def test_current_year_is_marked_as_part_of_one_before_it_is_projected():
    # 40 days into the year there is no projection yet. The bar is a ninth of
    # last year's, and the page says that it only runs to 9 February.
    df = clean_feed(make_raw_feed(end="2026-02-10 23:00"))
    at = run_page("2_Trends.py", make_page_state(df))
    yoy = next(f for f in plotly_figures(at) if f["data"][0]["type"] == "bar")
    [bars] = yoy["data"]  # no "Projected" bar on top of it
    assert values(bars["x"]) == [2024, 2025, 2026]
    assert values(bars["marker"]["opacity"]) == [0.45, 1, 0.45]
    assert [c.value for c in at.caption] == [
        "Lighter bars cover part of a year: 2024 (from 1 Jun), 2026 (to 9 Feb)."
    ]


def test_projection_counts_from_the_first_day_of_data():
    # The data starts on 1 February of the current year: at 8 a day the year
    # comes to 2,920, not to the 2,417 of a count taken to run from 1 January
    df = clean_feed(make_raw_feed(start="2026-02-01", end="2026-06-30 23:00"))
    at = run_page("2_Trends.py", make_page_state(df))
    yoy = next(f for f in plotly_figures(at) if f["data"][0]["type"] == "bar")
    actual, projected = yoy["data"]
    assert projected["base"][0] + projected["y"][0] == pytest.approx(8 * 365)
    assert [c.value for c in at.caption] == ["Lighter bars cover part of a year: 2026 (1 Feb to 29 Jun)."]


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


def test_pivot_says_how_many_incidents_it_cannot_place(page_state, clean_df):
    # Every incident has a category and a status: all are counted, nothing to add
    at = run_page("4_Analysis.py", page_state)
    assert int(at.dataframe[0].value.to_numpy().sum()) == len(clean_df)
    assert not at.caption

    # Seven incidents without a case status have no column to be counted in
    no_status = clean_df.index.isin(clean_df.index[:7])
    some_missing = clean_df.assign(case_status=clean_df["case_status"].mask(no_status))
    at = run_page("4_Analysis.py", dict(page_state, df_filtered=some_missing))
    assert not at.error
    assert int(at.dataframe[0].value.to_numpy().sum()) == len(clean_df) - 7
    assert [c.value for c in at.caption] == [
        f"7 of {len(clean_df):,} incidents have no offense_category or case_status and are not in the table."
    ]


def test_pivot_names_the_days_of_the_week(page_state):
    at = run_page("4_Analysis.py", page_state)
    at.selectbox(key="pivot_rows").select("incident_day_of_week").run()
    assert not at.exception and not at.error

    # Monday to Sunday, in that order, instead of a bare 1 to 7
    assert at.dataframe[0].value.index.tolist() == DAY_LABELS
    assert plotly_figures(at)[0]["layout"]["yaxis"]["type"] == "category"


def test_pivot_year_axis_ticks_on_whole_years(clean_df):
    at = run_page("4_Analysis.py", make_page_state(clean_df, year=2025))
    at.selectbox(key="pivot_columns").select("incident_year").run()
    assert not at.error
    figure = plotly_figures(at)[0]
    assert figure["layout"]["xaxis"]["dtick"] == 1
    assert at.dataframe[0].value.columns.tolist() == [2025]


def test_pivot_hours_stay_whole_numbers_when_a_row_has_none(raw_feed):
    # One empty cell makes read_csv hand over the whole hour column as floats
    feed = raw_feed.assign(incident_hour_of_day=raw_feed["incident_hour_of_day"].astype(float))
    feed.loc[feed.index[0], "incident_hour_of_day"] = np.nan
    df = clean_feed(feed)

    at = run_page("4_Analysis.py", dict(make_page_state(df), pivot_columns_selection="incident_hour_of_day"))
    assert not at.exception and not at.error
    # 0 to 23, not 0.0 to 23.0
    assert [str(hour) for hour in at.dataframe[0].value.columns] == [str(hour) for hour in range(24)]
    assert [c.value for c in at.caption] == [
        f"1 of {len(df):,} incidents have no offense_category or incident_hour_of_day and are not in the table."
    ]


# ── Geography ────────────────────────────────────────────────────────────────

def test_geography_charts(page_state):
    at = run_page("3_Geography.py", page_state)
    figures = plotly_figures(at)
    assert len(figures) == 5
    # Every precinct and district, not a top 10 that would leave one of the 11 precincts out
    precincts = figures[1]["data"][0]["y"]
    assert set(precincts) == {f"Precinct {n}" for n in range(2, 13)}
    districts = figures[2]["data"][0]["y"]
    assert set(districts) == {f"District {n}" for n in range(1, 8)}
    assert [s.value for s in at.subheader][1:3] == ["Police Precincts", "Council Districts"]
    zips = figures[3]["data"][0]["y"]
    assert set(zips) == {"48201", "48226", "48207", "48210"}


def test_geography_without_a_value_to_count(page_state, clean_df):
    # No incident in the selection has a neighborhood or a usable zip code
    nowhere = clean_df.assign(neighborhood=np.nan, zip_code="unknown")
    at = run_page("3_Geography.py", dict(page_state, df_filtered=nowhere))
    assert [i.value for i in at.info] == ["Nothing recorded for this selection."] * 2
    assert len(plotly_figures(at)) == 3


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


def test_map_legend_names_what_the_grey_points_are(page_state, clean_df):
    # Five incidents have no category: they are drawn in the grey that the
    # categories beyond the palette share, and the legend says so
    no_category = clean_df.index.isin(clean_df.index[:5])
    some_unknown = clean_df.assign(offense_category=clean_df["offense_category"].mask(no_category))
    legend = legend_colors(run_page("5_Map.py", dict(page_state, df_filtered=some_unknown)))
    assert "Other (5 categories) and Unknown" in legend
    assert len(legend) == 16 and len(set(legend.values())) == 16

    # A selection within the palette: the grey is there for the unknown ones alone
    category = page_state["category_order"][0]
    selection = some_unknown[no_category | (some_unknown["offense_category"] == category)]
    legend = legend_colors(run_page("5_Map.py", dict(page_state, df_filtered=selection)))
    assert list(legend) == [category, "Unknown"]
    assert legend["Unknown"] == "rgb(140,150,165)"

    # The grey points carry that name in their tooltip
    points = deck_spec(run_page("5_Map.py", dict(page_state, df_filtered=selection)))["layers"][0]["data"]
    assert {(p["r"], p["g"], p["b"]) for p in points if p["category"] == "Unknown"} == {(140, 150, 165)}


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


def test_map_view_is_kept_for_the_next_visit(page_state):
    at = run_page("5_Map.py", page_state)
    assert at.radio[0].value == "Scatter"
    at.radio[0].set_value("Heatmap").run()
    assert at.session_state["map_view_selection"] == "Heatmap"

    # Coming back to the page: widget state is gone, the saved choice is not
    at = run_page("5_Map.py", dict(page_state, map_view_selection="Heatmap"))
    assert at.radio[0].value == "Heatmap"
    assert deck_spec(at)["layers"][0]["@@type"] == "HeatmapLayer"


def test_map_heatmap_weights_cover_every_incident(page_state, clean_df):
    at = run_page("5_Map.py", page_state)
    at.radio[0].set_value("Heatmap").run()
    assert not at.exception

    layer = deck_spec(at)["layers"][0]
    assert layer["@@type"] == "HeatmapLayer"
    assert sum(point["weight"] for point in layer["data"]) == len(clean_df)


def test_map_heatmap_is_coarsened_to_stay_within_its_size_limit(page_state, clean_df, monkeypatch):
    import constants

    heatmap = dict(page_state, map_view_selection="Heatmap")
    spots = deck_spec(run_page("5_Map.py", heatmap))["layers"][0]["data"]
    assert len(spots) > 500

    # With more spots than may be sent, positions are rounded more coarsely
    monkeypatch.setattr(constants, "MAX_HEATMAP_CELLS", 500)
    coarser = deck_spec(run_page("5_Map.py", heatmap))["layers"][0]["data"]
    assert len(coarser) <= 500
    # Every incident still counts towards its spot
    assert sum(point["weight"] for point in coarser) == len(clean_df)
