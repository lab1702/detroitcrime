"""Page-level tests, run headlessly with Streamlit's AppTest."""

import base64
import gc
import json
import threading
import warnings
import weakref
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import requests
import streamlit as st
from streamlit.proto.Metric_pb2 import Metric as MetricProto
from streamlit.testing.v1 import AppTest
from urllib3.exceptions import ReadTimeoutError

import utils
from conftest import clean_feed, make_page_state, make_raw_feed
from constants import CHART_HEIGHT, DAY_LABELS, HOLIDAY_LABEL_DAYS_PER_CHARACTER, MAX_SCATTER_POINTS
from theme import KPI_ROW_KEY, PLOT_TOP_MARGIN

ROOT = Path(__file__).resolve().parent.parent
TIMEOUT = 300  # the Trends page fits a Prophet model

# What a page says while it waits for a download to end
REFRESHING_NOTE = "Newer data is being downloaded. This page will update when it is ready."
RETRY_NOTE = "The download will be tried again shortly, and this page will update by itself."
# The same after a failure that would only repeat, which the cache is slower to try again
LATER_NOTE = "The download will be tried again later, and this page will update by itself."


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


# The day the app is taken to be running on: two days after the synthetic
# feed ends, as the real export runs a few days behind. By the real date the
# feed would end long ago, and be refused as cut short.
TODAY = date(2026, 4, 2)


@pytest.fixture(autouse=True)
def today(monkeypatch):
    monkeypatch.setattr(utils, "local_today", lambda timezone=None: TODAY)


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


def test_pivot_page_is_handed_the_selection_by_the_app(raw_feed, clean_df, monkeypatch):
    # The other page tests hand a page its session state themselves. Here it
    # is app.py that does, so the two cannot come to disagree unnoticed.
    at = run_app(raw_feed, monkeypatch)
    at.sidebar.selectbox[1].select(2025).run()
    at.switch_page("views/4_Analysis.py").run()

    assert not at.exception, at.exception
    assert not at.error and not at.warning
    in_2025 = int((clean_df["incident_year"] == 2025).sum())
    assert int(at.dataframe[0].value.to_numpy().sum()) == in_2025


def test_map_page_is_handed_the_category_ranking_by_the_app(raw_feed, clean_df, monkeypatch):
    at = run_app(raw_feed, monkeypatch)
    at.switch_page("views/5_Map.py").run()
    assert not at.exception, at.exception
    assert not at.warning
    assert len(deck_spec(at)["layers"][0]["data"]) == len(clean_df)
    everything = legend_colors(at)
    assert "Other (5 categories)" in everything

    # Filtered to the fourth most frequent category, it keeps the color it has
    # in the full view. Without the ranking of the whole dataset from app.py
    # the page ranks the selection, and this one would turn the first color.
    category = clean_df["offense_category"].value_counts().index[3]
    at.sidebar.selectbox[0].select(category).run()
    assert not at.exception, at.exception
    assert "Map" in [h.value for h in at.header]
    assert legend_colors(at) == {category: everything[category]}


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


def keyed_block(at: AppTest, key: str):
    """The container the page gave `key`, wherever on the page it is."""
    def search(node):
        for child in getattr(node, "children", {}).values():
            if getattr(child, "key", None) == key:
                return child
            found = search(child)
            if found is not None:
                return found
        return None

    return search(at.main)


def test_kpi_cards_are_in_the_row_the_theme_lays_out(overview):
    # Four across in a window of laptop width, the cards cut their figures off
    # ("751,…"). The stylesheet lays the row out by the room it has, and finds
    # it by this key: all four cards have to be under it.
    row = keyed_block(overview, KPI_ROW_KEY)
    assert [m.label for m in row.metric] == ["Total Incidents", "Date Range", "Avg Daily", "Most Common"]


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


def frames_in_memory() -> list[pd.DataFrame]:
    return [found for found in gc.get_objects() if isinstance(found, pd.DataFrame)]


@pytest.mark.parametrize("page", [
    None, "views/2_Trends.py", "views/3_Geography.py", "views/4_Analysis.py", "views/5_Map.py",
], ids=["Overview", "Trends", "Geography", "Pivot Analysis", "Map"])
def test_page_lets_go_of_its_data_when_the_run_ends(raw_feed, monkeypatch, page):
    # An empty `run_data` is not the whole of it. A page that kept the data at
    # the top level of its script went on holding it there: the Pivot page for
    # as long as the tab stayed open (through the Swap button's callback), the
    # others until Python next looked for unreachable objects.
    at = run_app(raw_feed, monkeypatch)
    # One year: a selection of this tab's own, not the dataset every tab shares
    at.sidebar.selectbox[1].select(2025).run()
    if page:
        at.switch_page(page).run()
    assert not at.exception, at.exception

    gc.collect()
    there_before = [weakref.ref(frame) for frame in frames_in_memory()]
    gc.disable()  # what is let go of is let go of by the run ending, not by a collection
    try:
        at.run()
        assert not at.exception, at.exception
        left_behind = [
            frame.shape for frame in frames_in_memory()
            if not any(known() is frame for known in there_before)
        ]
    finally:
        gc.enable()
    assert left_behind == []


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
        "The app could not reach the data portal."
    ]
    assert not at.metric
    # A fault on the line can pass: the page says it will look again by itself
    assert [c.value for c in at.caption] == [RETRY_NOTE]


def test_passing_first_load_failure_is_tried_again_without_a_reload(raw_feed, monkeypatch):
    # The cache waits no time before another try, so the page is at once due to look again
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
    # It would say the same on every try, so nothing is promised "shortly".
    # The page still looks again, as seldom as the cache tries: a portal that
    # is republishing the dataset answers like this for a while.
    assert [c.value for c in at.caption] == [LATER_NOTE]


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
        "The app could not reach the data portal."
    ]


def test_download_cut_short_is_reported_in_plain_words(monkeypatch):
    def cut_short(url, **kwargs):
        raise requests.exceptions.ChunkedEncodingError("Connection broken: IncompleteRead(1024 bytes read)")

    monkeypatch.setattr(utils, "safe_load_data_from_url", cut_short)
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=TIMEOUT)
    at.run()
    assert [e.value for e in at.error] == ["The download from the data portal was interrupted before it finished."]


def test_failure_is_not_put_to_the_visitor_to_fix(monkeypatch):
    # The download is the app's: the visitor's own connection is fine (the page
    # reached them), and the page says it will try again by itself
    def timed_out(url, **kwargs):
        raise requests.exceptions.ReadTimeout("read timed out")

    monkeypatch.setattr(utils, "safe_load_data_from_url", timed_out)
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=TIMEOUT)
    at.run()
    assert [e.value for e in at.error] == ["The data portal took too long to answer."]
    assert [c.value for c in at.caption] == [RETRY_NOTE]


def test_portal_that_goes_silent_mid_download_is_not_reported_as_unreachable(monkeypatch):
    # `requests` does not call this one a timeout: it wraps urllib3's error in
    # a ConnectionError, which used to read "could not reach the data portal"
    def stalled(url, **kwargs):
        raise requests.exceptions.ConnectionError(ReadTimeoutError(None, url, "Read timed out."))

    monkeypatch.setattr(utils, "safe_load_data_from_url", stalled)
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=TIMEOUT)
    at.run()
    assert [e.value for e in at.error] == ["The data portal took too long to answer."]
    assert [c.value for c in at.caption] == [RETRY_NOTE]


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
        "The app could not reach the data portal."
    ]
    assert not overview.caption  # nothing is being downloaded any more
    assert {m.label for m in overview.metric} >= {"Total Incidents", "Avg Daily"}


def test_refreshed_data_is_shown_on_the_run_after_it_arrives(raw_feed, monkeypatch):
    # The copy in use was loaded three days ago; the feed has those days by
    # the time the data is due for a refresh
    earlier = raw_feed[raw_feed["incident_occurred_at"] < "2026/03/29"]
    overview = run_app(earlier, monkeypatch)
    download, release = held_back(raw_feed)
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

    def get_versioned(self, loader):
        self.gets += 1
        return self.df, None, self.gets

    def wait(self, timeout=None) -> bool:
        return not (self.still_running.pop(0) if self.still_running else False)

    def at_rest(self, version) -> bool:
        return self.wait()


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


def test_refresh_that_ends_as_the_page_starts_is_not_missed(clean_df, monkeypatch):
    # The refresh ended between the page being handed the old copy and its
    # asking about it: nothing is running any more, but what the page holds is
    # no longer what there is. It used to stay on the old figures.
    class JustEnded(WaitingCache):
        def at_rest(self, version) -> bool:
            return version > 1

    cache = JustEnded(clean_df, still_running=[])
    monkeypatch.setattr(utils, "dataset_cache", lambda: cache)
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=TIMEOUT)
    at.run()

    assert not at.exception, at.exception
    assert cache.gets == 2
    assert not at.caption
    assert at.metric


class RestingCache(WaitingCache):
    """Stands in for the dataset cache with nothing under way.

    `at_rest_answers` is what it says, in turn, each time it is asked whether
    it is still at rest; after that it is. It notes the version asked about.
    """

    def __init__(self, df: pd.DataFrame, at_rest_answers: list[bool]):
        super().__init__(df, still_running=[])
        self.at_rest_answers = at_rest_answers
        self.asked_about = []

    def at_rest(self, version) -> bool:
        self.asked_about.append(version)
        return self.at_rest_answers.pop(0) if self.at_rest_answers else True


def test_page_left_open_is_drawn_again_once_the_cache_has_moved_on(clean_df, monkeypatch):
    # Nothing to wait for when the page is drawn. Afterwards another visitor
    # sets a refresh off, or the data comes due: the page asks now and then
    # (here it is on its first time of asking), and used to keep its figures.
    cache = RestingCache(clean_df, at_rest_answers=[True, False])
    monkeypatch.setattr(utils, "dataset_cache", lambda: cache)
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=TIMEOUT)
    at.run()

    assert not at.exception, at.exception
    assert cache.gets == 2
    # It asked about the copy it was handed each time, and says nothing while it waits
    assert cache.asked_about == [1, 1, 2, 2]
    assert not at.caption
    assert at.metric


def test_page_left_open_stays_as_it_is_while_the_cache_is_at_rest(clean_df, monkeypatch):
    cache = RestingCache(clean_df, at_rest_answers=[])
    monkeypatch.setattr(utils, "dataset_cache", lambda: cache)
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=TIMEOUT)
    at.run()

    assert not at.exception, at.exception
    assert cache.gets == 1
    assert not at.caption
    assert at.metric


class FailedOnce(RestingCache):
    """Stands in for the dataset cache after a first load that failed with `cause`.

    The first run that asks is told of the failure, at version 7; the next is
    given the data.
    """

    def __init__(self, df: pd.DataFrame, at_rest_answers: list[bool], cause: Exception):
        super().__init__(df, at_rest_answers)
        self.cause = cause

    def get_versioned(self, loader):
        if self.gets == 0:
            self.gets += 1
            raise utils.LoadFailedError(str(self.cause), version=7) from self.cause
        return super().get_versioned(loader)


def test_failed_first_load_is_tried_again_when_the_cache_says_it_is_due(clean_df, monkeypatch):
    # The page arrives late in the cache's wait. It goes back for the data
    # when the cache is due another try, not a whole wait after it was opened.
    portal_is_down = requests.exceptions.ConnectionError("portal down")

    # At rest when the page is drawn...
    cache = FailedOnce(clean_df, at_rest_answers=[True], cause=portal_is_down)
    monkeypatch.setattr(utils, "dataset_cache", lambda: cache)
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=TIMEOUT)
    at.run()
    assert [e.value for e in at.error] == ["The app could not reach the data portal."]
    assert [c.value for c in at.caption] == [RETRY_NOTE]
    assert (cache.gets, cache.asked_about) == (1, [7])

    # ...and due another try by the time a page asks again
    cache = FailedOnce(clean_df, at_rest_answers=[False], cause=portal_is_down)
    monkeypatch.setattr(utils, "dataset_cache", lambda: cache)
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=TIMEOUT)
    at.run()
    assert not at.exception, at.exception
    assert cache.gets == 2
    assert not at.error and not at.caption
    assert at.metric


def test_open_page_goes_back_for_the_data_after_a_failure_that_looked_set_to_repeat(clean_df, monkeypatch):
    # An export caught half written cannot be parsed, and the same file never
    # will be; the next export is whole. A page left open used to stay on the
    # failure until someone reloaded it.
    half_written = pd.errors.ParserError("Error tokenizing data. C error: EOF inside string starting at row 5")
    assert not utils.is_transient_error(half_written)

    # At rest when the page is drawn: it says it will look again, though not shortly...
    cache = FailedOnce(clean_df, at_rest_answers=[True], cause=half_written)
    monkeypatch.setattr(utils, "dataset_cache", lambda: cache)
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=TIMEOUT)
    at.run()
    assert [e.value for e in at.error] == [
        "Data validation error: Error tokenizing data. C error: EOF inside string starting at row 5"
    ]
    assert [c.value for c in at.caption] == [LATER_NOTE]
    assert not at.metric
    # It asks about the failure it was told of, and no sooner goes back for the data
    assert (cache.gets, cache.asked_about) == (1, [7])

    # ...and goes back once the cache is due another try
    cache = FailedOnce(clean_df, at_rest_answers=[False], cause=half_written)
    monkeypatch.setattr(utils, "dataset_cache", lambda: cache)
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=TIMEOUT)
    at.run()
    assert not at.exception, at.exception
    assert cache.gets == 2
    assert not at.error and not at.caption
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
    # What to do about it is for whoever runs the app, and goes to the log:
    # the page asks nothing of a visitor
    assert "restart" not in warning


def test_refresh_that_ends_days_earlier_is_refused(overview, raw_feed, monkeypatch):
    before = {m.label: m.value for m in overview.metric}

    # An export cut short by date, five days before the copy in use ends
    # (local midnight on 27 Mar is 04:00 UTC). It has all but 40 of its 5,344
    # incidents: too few are gone for it to be refused as shrunken.
    cut_short = raw_feed[raw_feed["incident_occurred_at"] < "2026/03/27 04:00:00+00"]
    monkeypatch.setattr(utils, "safe_load_data_from_url", lambda url, **kwargs: cut_short.copy())
    utils.dataset_cache().expire()
    overview.run()
    assert utils.dataset_cache().wait(timeout=TIMEOUT)
    overview.run()
    assert not overview.exception and not overview.error

    assert {m.label: m.value for m in overview.metric} == before
    assert [w.value for w in overview.warning] == [
        "The data could not be refreshed, so the last copy that loaded is shown. "
        "Data validation error: the refreshed feed ends on 2026-03-25, 5 days before the copy shown; "
        "it may be incomplete"
    ]


def test_first_load_of_a_feed_that_ends_long_ago_is_refused(raw_feed, monkeypatch):
    # An export cut short by date: everything but the last two months. It has
    # nine in ten of its incidents, and with no earlier copy to hold it
    # against it used to be shown as the whole dataset.
    cut_short = raw_feed[raw_feed["incident_occurred_at"] < "2026/02/01"]
    monkeypatch.setattr(utils, "safe_load_data_from_url", lambda url, **kwargs: cut_short.copy())
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=TIMEOUT)
    at.run()
    assert [e.value for e in at.error] == [
        "Data validation error: the feed ends on 2026-01-30, 62 days ago; it may be incomplete, "
        "or the portal may have stopped updating it"
    ]
    assert not at.metric
    # The same export would be refused again, at the cost of a whole download:
    # the page tries again no sooner than the cache will
    assert [c.value for c in at.caption] == [LATER_NOTE]

    # The whole export is taken once the wait before another try is over
    monkeypatch.setattr(utils, "safe_load_data_from_url", lambda url, **kwargs: raw_feed.copy())
    utils.dataset_cache().expire()
    at.run()
    assert not at.error
    assert at.metric


def test_feed_a_few_weeks_behind_is_shown(raw_feed, clean_df, monkeypatch):
    # The export runs behind, at times by more than its usual few days
    monkeypatch.setattr(utils, "local_today", lambda timezone=None: date(2026, 4, 29))
    at = run_app(raw_feed, monkeypatch)
    assert not at.error
    assert {m.label: m.value for m in at.metric}["Total Incidents"] == f"{len(clean_df):,}"


def test_filters_are_kept_when_a_refresh_changes_their_options(raw_feed, monkeypatch):
    # A feed with only the last two days of 2024 (local midnight on 30 Dec is
    # 05:00 UTC): few enough that a refresh without them is not refused as shrunken
    first = raw_feed[raw_feed["incident_occurred_at"] >= "2024/12/30 05:00:00+00"]
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
    overview.sidebar.selectbox[0].select("CATEGORY 20").run()
    overview.sidebar.selectbox[1].select(2025).run()

    # The category has gone from the refreshed feed (the rarest, well under one
    # incident in a hundred: a feed that had lost more would be refused as incomplete)
    without = raw_feed[raw_feed["offense_category"] != "CATEGORY 20"]
    monkeypatch.setattr(utils, "safe_load_data_from_url", lambda url, **kwargs: without.copy())
    utils.dataset_cache().expire()
    overview.run()
    assert utils.dataset_cache().wait(timeout=TIMEOUT)
    overview.run()
    assert not overview.exception

    category, year = overview.sidebar.selectbox
    assert "CATEGORY 20" not in category.options
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
    # each time: the page tries again no sooner than the cache will
    assert [c.value for c in at.caption] == [LATER_NOTE]

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
    # The page says that one incident is not in the chart
    assert f"1 of {len(df):,} incidents have no usable day or hour and are not in the chart." in [
        c.value for c in at.caption
    ]


def test_heatmap_says_how_many_incidents_it_has_no_cell_for(raw_feed):
    # Day 0 and hour 24 are numbers, but not a day of the week or an hour of the day
    feed = raw_feed.copy()
    feed.loc[feed.index[:50], "incident_day_of_week"] = 0
    feed.loc[feed.index[50:90], "incident_hour_of_day"] = 24
    df = clean_feed(feed)
    odd = int(((df["incident_day_of_week"] == 0) | (df["incident_hour_of_day"] == 24)).sum())
    assert odd > 0

    at = run_page("2_Trends.py", make_page_state(df))
    heatmap = next(f for f in plotly_figures(at) if f["data"][0]["type"] == "heatmap")["data"][0]
    assert sum(values(heatmap["z"])) == len(df) - odd
    assert f"{odd} of {len(df):,} incidents have no usable day or hour and are not in the chart." in [
        c.value for c in at.caption
    ]


def test_heatmap_with_every_incident_in_it_has_no_note(page_state):
    at = run_page("2_Trends.py", page_state)
    assert not any("day or hour" in c.value for c in at.caption)


def test_forecast_names_each_holiday_once_and_in_full(page_state):
    at = run_page("2_Trends.py", page_state)
    # In the order of their days (they are drawn in another: see the next test)
    notes = sorted(forecast_figure(at)["layout"]["annotations"], key=lambda note: note["x"])
    names = [note["text"] for note in notes]
    # The year after 30 March 2026. 4 July 2026 is a Saturday: the Friday it is
    # observed on carries no second "Independence Day" next to the first.
    assert names == [
        "Memorial Day", "Juneteenth", "Independence Day", "Labor Day", "Columbus Day", "Veterans Day",
        "Thanksgiving Day", "Christmas Day", "New Year's Day", "MLK Day", "Presidents' Day",
    ]


def test_holiday_names_on_the_forecast_do_not_run_into_each_other(page_state):
    at = run_page("2_Trends.py", page_state)
    layout = forecast_figure(at)["layout"]
    notes = layout["annotations"]

    # Each name stands over its own line at the top of the plot, wherever the
    # forecast runs: set by its values, the levels closed up on one another
    # when it ran flat
    assert {(note["yref"], note["y"], note["ax"]) for note in notes} == {("paper", 1, 0)}

    # Two names on one level are further apart than they are wide. Thanksgiving
    # and Presidents' Day, 81 days apart, used to share one (every fourth name
    # did) and were drawn across each other in a window of laptop width.
    def days_wide(note: dict) -> int:
        return (len(note["text"]) + 3) * HOLIDAY_LABEL_DAYS_PER_CHARACTER

    by_day = sorted(notes, key=lambda note: note["x"])
    for i, left in enumerate(by_day):
        for right in by_day[i + 1:]:
            if left["ay"] == right["ay"]:
                apart = (pd.Timestamp(right["x"]) - pd.Timestamp(left["x"])).days
                assert apart >= (days_wide(left) + days_wide(right)) / 2, (left["text"], right["text"])
    level = {note["text"]: note["ay"] for note in notes}
    assert level["Thanksgiving Day"] != level["Presidents' Day"]
    # Names with room beside them are not stacked for nothing
    assert level["Memorial Day"] == level["Labor Day"] == max(level.values())

    # The highest are drawn first: the arrow from a name to its line then runs
    # behind the names below it, not across them
    assert [note["ay"] for note in notes] == sorted(note["ay"] for note in notes)

    # There is room above the plot for the highest of them (a name is 14 high),
    # and the plot itself is as tall as on the other charts
    assert layout["margin"]["t"] >= -min(level.values()) + 7
    assert layout["height"] - layout["margin"]["t"] == CHART_HEIGHT - PLOT_TOP_MARGIN
    # Only the forecast has that room: the other charts have what every chart has
    others = [figure["layout"] for figure in plotly_figures(at) if "annotations" not in figure["layout"]]
    assert [other["margin"]["t"] for other in others] == [PLOT_TOP_MARGIN] * 3


def test_holiday_on_the_last_day_of_data_is_not_marked_on_the_forecast():
    # The feed's last day, 28 November 2025, is left out as incomplete, so the
    # data ends on Thanksgiving Day: history, like Labor Day in the months
    # shown before it, which has no mark either
    df = clean_feed(make_raw_feed(end="2025-11-28 23:00"))
    assert df["date"].max() == date(2025, 11, 27)

    at = run_page("2_Trends.py", make_page_state(df))
    notes = forecast_figure(at)["layout"]["annotations"]
    # Only the Thanksgiving that is forecast, a year on
    assert [note["text"] for note in notes].count("Thanksgiving Day") == 1
    assert min(note["x"][:10] for note in notes) > "2025-11-27"


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
        "Fainter bars cover part of a year: 2024 (from 1 Jun), 2026 (to 30 Mar)."
    ]

    # A whole year on its own is neither fainter nor explained
    at = run_page("2_Trends.py", make_page_state(page_state["df_all_years"], year=2025))
    yoy = next(f for f in plotly_figures(at) if f["data"][0]["type"] == "bar")
    assert values(yoy["data"][0]["marker"]["opacity"]) == [1]
    assert not any("Fainter bars" in c.value for c in at.caption)


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
        "Fainter bars cover part of a year: 2024 (from 1 Jun), 2026 (to 9 Feb)."
    ]


def test_projection_counts_from_the_first_day_of_data():
    # The data starts on 1 February of the current year: at 8 a day the year
    # comes to 2,920, not to the 2,417 of a count taken to run from 1 January
    df = clean_feed(make_raw_feed(start="2026-02-01", end="2026-06-30 23:00"), today=date(2026, 7, 2))
    at = run_page("2_Trends.py", make_page_state(df))
    yoy = next(f for f in plotly_figures(at) if f["data"][0]["type"] == "bar")
    actual, projected = yoy["data"]
    assert projected["base"][0] + projected["y"][0] == pytest.approx(8 * 365)
    assert [c.value for c in at.caption] == ["Fainter bars cover part of a year: 2026 (1 Feb to 29 Jun)."]


def test_forecast_failure_is_reported_and_not_cached(page_state, monkeypatch, caplog):
    from prophet import Prophet

    def broken_fit(self, *args, **kwargs):
        raise RuntimeError("optimizer exploded")

    with monkeypatch.context() as patched:
        patched.setattr(Prophet, "fit", broken_fit)
        at = run_page("2_Trends.py", page_state)
        assert [e.value for e in at.error] == ["Error creating forecast: optimizer exploded"]
        assert not at.info  # not blamed on "not enough data"
        # The page shows the message only: the log says where it went wrong
        [record] = [r for r in caplog.records if r.name == "views.trends"]
        assert record.levelname == "ERROR"
        assert "RuntimeError: optimizer exploded" in caplog.text
        assert "in broken_fit" in caplog.text

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


def test_pivot_failure_is_reported_and_logged(page_state, monkeypatch, caplog):
    def broken_pivot(*args, **kwargs):
        raise RuntimeError("pivot exploded")

    monkeypatch.setattr(pd, "pivot_table", broken_pivot)
    at = run_page("4_Analysis.py", page_state)
    assert [e.value for e in at.error] == ["Error creating pivot table: pivot exploded"]
    assert not at.dataframe
    # The page shows the message only: the log says where it went wrong
    [record] = [r for r in caplog.records if r.name == "views.analysis"]
    assert record.levelname == "ERROR"
    assert "RuntimeError: pivot exploded" in caplog.text
    assert "in broken_pivot" in caplog.text


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


def test_geography_without_a_value_to_count(raw_feed):
    # No incident has a neighborhood or a usable zip code
    nowhere = clean_feed(raw_feed.assign(neighborhood=np.nan, zip_code="unknown"))
    at = run_page("3_Geography.py", make_page_state(nowhere))
    assert [i.value for i in at.info] == ["Nothing recorded for this selection."] * 2
    assert len(plotly_figures(at)) == 3


def test_incidents_without_a_valid_code_are_left_out_where_codes_are_counted(raw_feed):
    # One zip code written with its extension makes read_csv hand over the
    # whole column as text. The codes are read once, when the data is loaded.
    feed = raw_feed.assign(zip_code=raw_feed["zip_code"].astype(str))
    feed.loc[feed.index[:40], "zip_code"] = "48201-1234"
    df = clean_feed(feed)
    odd = int(df["zip_code"].isna().sum())
    assert 0 < odd <= 40

    at = run_page("3_Geography.py", make_page_state(df))
    zips = plotly_figures(at)[3]["data"][0]
    assert set(zips["y"]) == {"48201", "48226", "48207", "48210"}
    assert sum(values(zips["x"])) == len(df) - odd

    # The pivot counts the same incidents, and says how many it has no cell for
    at = run_page("4_Analysis.py", dict(make_page_state(df), pivot_rows_selection="zip_code"))
    assert not at.error
    table = at.dataframe[0].value
    assert table.index.tolist() == ["48201", "48207", "48210", "48226"]
    assert int(table.to_numpy().sum()) == len(df) - odd
    assert [c.value for c in at.caption] == [
        f"{odd} of {len(df):,} incidents have no zip_code or case_status and are not in the table."
    ]


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


def test_map_with_a_column_that_is_empty_throughout(page_state, clean_df):
    # A column with nothing in it is read as numbers. Filling its gaps with
    # text is deprecated in pandas, and an error from pandas 4 on.
    empty = clean_df.assign(offense_category=np.nan, nearest_intersection=np.nan)
    with warnings.catch_warnings():
        warnings.filterwarnings("error", message=".*not supported as a fill value")
        at = run_page("5_Map.py", dict(page_state, df_filtered=empty))

    points = deck_spec(at)["layers"][0]["data"]
    assert len(points) == len(clean_df)
    assert {(p["category"], p["location"]) for p in points} == {("Unknown", "Unknown location")}


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
