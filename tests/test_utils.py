"""Data loading, cleaning and year-over-year maths."""

import threading
from datetime import date

import pandas as pd
import plotly.express as px
import pytest
import requests

import utils
from constants import REQUIRED_COLUMNS
from theme import BLUE_ACCENT, RED_ACCENT, apply_dark_plotly
from utils import (
    DataNotReadyError, RefreshingCache, clean_and_filter_data, code_labels, compute_yoy_deltas,
    daily_incident_counts, format_categorical_series, numeric_codes, project_full_year,
    safe_load_data_from_url, selection_period,
)

BOUNDS = ((42.0, 43.0), (-84.0, -82.0))


def feed_rows(timestamps, **overrides) -> pd.DataFrame:
    """Minimal raw rows for the given UTC timestamps."""
    n = len(timestamps)
    data = {
        "latitude": [42.33] * n,
        "longitude": [-83.04] * n,
        "incident_occurred_at": timestamps,
        "offense_category": ["BURGLARY"] * n,
        "offense_description": ["BURGLARY - FORCED ENTRY"] * n,
        "case_status": ["ACTIVE"] * n,
        "neighborhood": ["Greektown"] * n,
        "nearest_intersection": ["Brush St & Monroe St"] * n,
    }
    data.update(overrides)
    return pd.DataFrame(data)


# A later row so the incidents under test are not on the dropped "most recent" day
LATER = "2030/01/01 12:00:00+00"
# "Today" for these tests: after every timestamp they use, LATER included
TODAY = date(2031, 1, 1)


def clean(df: pd.DataFrame, today: date = TODAY) -> pd.DataFrame:
    return clean_and_filter_data(df, "2018-01-01", *BOUNDS, today=today)


class TestLocalTime:
    def test_evening_incident_stays_on_its_local_day(self):
        # 03:33 UTC on 30 Nov is 22:33 on 29 Nov in Detroit (real row from the feed)
        df = clean(feed_rows(["2019/11/30 03:33:00+00", LATER]))
        assert df["date"].tolist() == [date(2019, 11, 29)]
        assert df["incident_occurred_at"].iloc[0] == pd.Timestamp("2019-11-29 22:33:00")

    def test_new_years_eve_stays_in_its_year(self):
        # 01:00 UTC on 1 Jan 2024 is 20:00 on 31 Dec 2023 in Detroit
        df = clean(feed_rows(["2024/01/01 01:00:00+00", LATER]))
        assert df["incident_year"].tolist() == [2023]
        assert df["date"].tolist() == [date(2023, 12, 31)]

    def test_daylight_saving_offset(self):
        # Summer: UTC-4, so 03:30 UTC is 23:30 the previous evening
        df = clean(feed_rows(["2024/07/10 03:30:00+00", LATER]))
        assert df["date"].tolist() == [date(2024, 7, 9)]

    def test_min_date_is_applied_in_local_time(self):
        # 02:00 UTC on 1 Jan 2018 is still 31 Dec 2017 in Detroit: before the cut
        df = clean(feed_rows(["2018/01/01 02:00:00+00", "2018/01/01 06:00:00+00", LATER]))
        assert df["date"].tolist() == [date(2018, 1, 1)]
        assert df["incident_year"].min() == 2018

    def test_most_recent_day_is_excluded(self):
        df = clean(feed_rows(["2024/03/01 15:00:00+00", "2024/03/02 15:00:00+00"]))
        assert df["date"].tolist() == [date(2024, 3, 1)]

    def test_year_matches_the_feed_for_synthetic_data(self, clean_df):
        assert (clean_df["incident_year"] == clean_df["incident_occurred_at"].dt.year).all()
        assert clean_df["incident_occurred_at"].dt.tz is None


class TestBadDates:
    def test_unreadable_timestamp_is_dropped_without_failing_the_load(self):
        good = [f"2024/03/{day:02d} 15:00:00+00" for day in range(1, 29)] * 8
        df = clean(feed_rows(good + ["not a date", LATER]))
        assert len(df) == len(good)

    def test_timestamps_written_differently_are_still_read(self):
        stamps = [
            "2024/03/01 15:00:00+00",
            "2024/03/02 15:00:00.250+00",  # fractional seconds
            "2024-03-03T15:00:00Z",        # ISO 8601
            "2024/03/05",                  # date only: midnight UTC, the evening before in Detroit
            LATER,
        ]
        df = clean(feed_rows(stamps))
        assert df["date"].tolist() == [date(2024, 3, 1), date(2024, 3, 2), date(2024, 3, 3), date(2024, 3, 4)]

    def test_a_feed_of_unreadable_timestamps_is_refused(self):
        with pytest.raises(ValueError, match="2 of 4 timestamps could not be read"):
            clean(feed_rows(["2024/03/01 15:00:00+00", "soon", "later", LATER]))

    def test_future_dated_rows_are_dropped(self):
        # Two mistyped future dates in a feed that really ends on 20 Nov 2026
        stamps = [
            "2026/11/18 15:00:00+00", "2026/11/19 15:00:00+00", "2026/11/20 15:00:00+00",
            "2026/12/25 15:00:00+00", "2027/06/01 15:00:00+00",
        ]
        df = clean(feed_rows(stamps), today=date(2026, 11, 20))
        # 20 Nov is the real most recent day, so that is the one left out as incomplete
        assert df["date"].tolist() == [date(2026, 11, 18), date(2026, 11, 19)]
        assert df["incident_year"].unique().tolist() == [2026]

    def test_today_defaults_to_the_current_date(self):
        now = pd.Timestamp.now(tz="UTC")
        stamps = [
            (now + pd.Timedelta(days=offset)).strftime("%Y/%m/%d %H:%M:%S+00")
            for offset in (-5, -2, 400)
        ]
        df = clean_and_filter_data(feed_rows(stamps), "2018-01-01", *BOUNDS)
        # The row dated next year goes, then the most recent real day
        assert len(df) == 1


class TestTextCleanup:
    def test_padded_descriptions_are_one_value(self):
        df = clean(feed_rows(
            ["2024/03/01 15:00:00+00", "2024/03/01 16:00:00+00", LATER],
            offense_description=["BURGLARY - FORCED ENTRY", "BURGLARY - FORCED ENTRY      ", "X"],
        ))
        assert df["offense_description"].value_counts().to_dict() == {"BURGLARY - FORCED ENTRY": 2}

    def test_missing_text_survives(self):
        df = clean(feed_rows(
            ["2024/03/01 15:00:00+00", LATER], neighborhood=[None, "Nolan"],
        ))
        assert df["neighborhood"].isna().tolist() == [True]


class TestPeriods:
    DATA = (date(2018, 1, 1), date(2026, 3, 30))

    def test_all_years_is_everything_the_data_covers(self):
        assert selection_period("All", *self.DATA) == self.DATA

    def test_a_year_is_that_calendar_year_cut_to_the_data(self):
        assert selection_period(2025, *self.DATA) == (date(2025, 1, 1), date(2025, 12, 31))
        assert selection_period(2026, *self.DATA) == (date(2026, 1, 1), date(2026, 3, 30))
        assert selection_period(2018, date(2018, 2, 3), date(2026, 3, 30)) == (date(2018, 2, 3), date(2018, 12, 31))

    def test_daily_counts_cover_every_day_of_the_period(self):
        # Three incidents on two days, in a period that starts before and ends after them
        df = pd.DataFrame({"date": [date(2025, 7, 10), date(2025, 7, 10), date(2025, 7, 12)]})
        daily = daily_incident_counts(df, date(2025, 7, 8), date(2025, 7, 14))
        assert daily["date"].tolist() == list(pd.date_range("2025-07-08", "2025-07-14"))
        assert daily["count"].tolist() == [0, 0, 2, 0, 1, 0, 0]


class TestYearOverYear:
    def test_projection_uses_days_of_data(self):
        # 59 days of data at 10 a day projects to a full year at the same rate
        assert project_full_year(590, date(2025, 2, 28)) is None  # under 60 days
        assert project_full_year(600, date(2025, 3, 1)) == pytest.approx(10 * 365)
        assert project_full_year(600, date(2024, 2, 29)) == pytest.approx(10 * 366)

    @staticmethod
    def yearly(prior_dates, current_dates) -> pd.DataFrame:
        dates = pd.to_datetime(list(prior_dates) + list(current_dates))
        return pd.DataFrame({"incident_year": dates.year, "date": dates.date})

    def test_flat_year_shows_no_change_whatever_today_is(self):
        prior = pd.date_range("2025-01-01", "2025-12-31", freq="D")
        current = pd.date_range("2026-01-01", "2026-03-06", freq="D")  # data ends 6 Mar
        incidents, avg = compute_yoy_deltas(self.yearly(prior, current), date(2026, 3, 6))
        assert incidents == "+0.0%"
        assert avg == "+0.0%"

    def test_sparse_prior_year_deltas_agree(self):
        # 2 incidents on consecutive days last year, 20 by day 300 this year
        prior = ["2025-03-03", "2025-03-04"]
        current = pd.date_range("2026-01-01", periods=20, freq="15D")
        incidents, avg = compute_yoy_deltas(self.yearly(prior, current), date(2026, 10, 27))
        assert incidents == avg == "+1116.7%"

    def test_no_delta_without_both_years(self):
        current = pd.date_range("2026-01-01", "2026-06-30", freq="D")
        assert compute_yoy_deltas(self.yearly([], current), date(2026, 6, 30)) == (None, None)


class TestCategoricalFormatting:
    def test_series_drops_invalid_and_adds_prefix(self):
        codes = pd.Series([3.0, 11, "07", None, 0, "n/a"], index=list("abcdef"))
        assert format_categorical_series(codes, "Precinct").to_dict() == {
            "a": "Precinct 3", "b": "Precinct 11", "c": "Precinct 7",
        }
        assert format_categorical_series(pd.Series([48226.0, None])).tolist() == ["48226"]

    def test_codes_are_numbers_with_zero_for_anything_invalid(self):
        assert numeric_codes(pd.Series([3.0, "07", None, "n/a", 0, "48201-1234"])).tolist() == [3, 7, 0, 0, 0, 0]

    def test_labels_keep_the_order_of_the_codes(self):
        assert code_labels([2, 9, 10], "Precinct") == ["Precinct 2", "Precinct 9", "Precinct 10"]
        assert code_labels([48201, 48226]) == ["48201", "48226"]


CSV_BODY = b"latitude,offense_category\n42.3,ARSON\n42.4,FRAUD\n"


class FakeResponse:
    """Stands in for a streamed `requests` response."""

    def __init__(self, body: bytes = CSV_BODY, status: int = 200, content_type: str = "text/csv",
                 drops: bool = False):
        self.body = body
        self.status_code = status
        self.headers = {"Content-Type": content_type}
        self.drops = drops

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.exceptions.HTTPError(f"{self.status_code} error", response=self)

    def iter_content(self, chunk_size):
        for i in range(0, len(self.body), 7):  # small, uneven chunks
            yield self.body[i:i + 7]
            if self.drops:
                raise requests.exceptions.ChunkedEncodingError("connection broken: IncompleteRead")


@pytest.fixture
def portal(monkeypatch):
    """Queue what the portal does on each download attempt; retries do not really wait.

    Each item is a FakeResponse to return or an exception to raise. Items are
    removed as they are used, so what is left shows how many attempts were made.
    """
    answers = []

    def fake_get(*args, **kwargs):
        answer = answers.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return answer

    monkeypatch.setattr(utils.requests, "get", fake_get)
    monkeypatch.setattr(utils.time, "sleep", lambda seconds: None)
    return answers


URL = "https://example.invalid/feed.csv"


class TestDownload:
    def test_reads_only_wanted_columns_from_a_bom_csv(self, portal):
        body = "﻿X,latitude,offense_category,ESRI_OID\n1,42.3,ARSON,9\n2,42.4,FRAUD,8\n".encode("utf-8")
        portal.append(FakeResponse(body))

        df = safe_load_data_from_url(URL, usecols=REQUIRED_COLUMNS)

        # Columns the feed lacks are left for validate_data to report
        assert df.columns.tolist() == ["latitude", "offense_category"]
        assert df["offense_category"].tolist() == ["ARSON", "FRAUD"]

    def test_loads_everything_without_usecols(self, portal):
        portal.append(FakeResponse(b"a,b\n1,2\n"))
        assert safe_load_data_from_url(URL).columns.tolist() == ["a", "b"]

    @pytest.mark.parametrize("failure", [
        FakeResponse(status=503),
        FakeResponse(status=429),
        requests.exceptions.ConnectTimeout("no answer"),
        requests.exceptions.ConnectionError("connection reset"),
        FakeResponse(drops=True),
    ], ids=["503", "429", "timeout", "connection error", "dropped mid-download"])
    def test_passing_failures_are_retried(self, portal, failure):
        portal.extend([failure, failure, FakeResponse()])
        df = safe_load_data_from_url(URL)
        assert df["offense_category"].tolist() == ["ARSON", "FRAUD"]
        assert portal == []

    def test_gives_up_after_max_retries(self, portal):
        portal.extend([FakeResponse(status=503)] * 4)
        with pytest.raises(requests.exceptions.HTTPError):
            safe_load_data_from_url(URL, max_retries=3)
        assert len(portal) == 1  # three attempts, no more

    def test_missing_file_is_not_retried(self, portal):
        portal.extend([FakeResponse(status=404), FakeResponse()])
        with pytest.raises(requests.exceptions.HTTPError):
            safe_load_data_from_url(URL)
        assert len(portal) == 1

    def test_unparseable_file_is_not_downloaded_again(self, portal):
        portal.extend([FakeResponse(b""), FakeResponse()])
        with pytest.raises(pd.errors.EmptyDataError):
            safe_load_data_from_url(URL)
        assert len(portal) == 1

    def test_status_document_instead_of_csv_is_not_ready(self, portal):
        status = FakeResponse(b'{"status":"ExportingData","progressInPercent":40}', content_type="application/json")
        portal.extend([status, status])
        with pytest.raises(DataNotReadyError, match="application/json instead of the CSV"):
            safe_load_data_from_url(URL, max_retries=2)
        assert portal == []  # worth waiting for, so it was tried again

    def test_accepted_but_not_ready_is_tried_again(self, portal):
        portal.extend([FakeResponse(b"", status=202), FakeResponse()])
        assert len(safe_load_data_from_url(URL)) == 2

    def test_needs_at_least_one_attempt(self, portal):
        with pytest.raises(ValueError, match="max_retries"):
            safe_load_data_from_url(URL, max_retries=0)


class Clock:
    """A clock the test moves by hand."""

    def __init__(self):
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


def loader(*results):
    """A loader that returns, or raises, each of `results` in turn."""
    remaining = list(results)

    def load():
        result = remaining.pop(0)
        if isinstance(result, Exception):
            raise result
        return result

    return load


class TestRefreshingCache:
    @pytest.fixture
    def clock(self) -> Clock:
        return Clock()

    @pytest.fixture
    def cache(self, clock) -> RefreshingCache:
        return RefreshingCache(ttl=100, retry_interval=10, clock=clock)

    def test_value_is_reused_until_it_is_due(self, cache, clock):
        load = loader("first", "second")
        assert cache.get(load) == ("first", None)
        clock.now += 99
        assert cache.get(load) == ("first", None)
        clock.now += 1
        assert cache.get(load) == ("second", None)

    def test_failed_refresh_keeps_the_previous_value(self, cache, clock):
        down = RuntimeError("portal down")
        load = loader("first", down, "second")
        cache.get(load)

        clock.now += 100
        assert cache.get(load) == ("first", down)
        # Not tried again until the retry interval has passed
        clock.now += 9
        assert cache.get(load) == ("first", down)
        clock.now += 1
        assert cache.get(load) == ("second", None)

    def test_first_load_failure_is_raised_and_nothing_is_kept(self, cache):
        load = loader(RuntimeError("portal down"), "first")
        with pytest.raises(RuntimeError, match="portal down"):
            cache.get(load)
        assert cache.get(load) == ("first", None)

    def test_other_callers_get_the_old_value_while_a_refresh_runs(self, cache, clock):
        cache.get(loader("first"))
        clock.now += 100
        seen = []

        def slow_refresh():
            # A second session asks for the data while this refresh is under way
            other = threading.Thread(target=lambda: seen.append(cache.get(loader())))
            other.start()
            other.join(timeout=5)
            return "second"

        assert cache.get(slow_refresh) == ("second", None)
        assert seen == [("first", None)]

    def test_expire_makes_the_next_get_refresh(self, cache):
        load = loader("first", "second")
        cache.get(load)
        cache.expire()
        assert cache.get(load) == ("second", None)


class TestAccentColors:
    def test_bars_and_lines_take_the_accent(self):
        bar = apply_dark_plotly(px.bar(x=[1, 2], y=["a", "b"], orientation="h"), RED_ACCENT)
        assert bar.data[0].marker.color == RED_ACCENT

        line = apply_dark_plotly(px.line(x=[1, 2], y=[3, 4]), BLUE_ACCENT)
        assert line.data[0].line.color == BLUE_ACCENT

    def test_explicit_colors_are_kept(self):
        fig = px.bar(x=[2025, 2026], y=[10, 4])
        fig.add_bar(x=[2026], y=[6], base=[4], marker_color="rgba(255, 165, 0, 0.3)")
        apply_dark_plotly(fig, BLUE_ACCENT)
        assert fig.data[0].marker.color == BLUE_ACCENT
        assert fig.data[1].marker.color == "rgba(255, 165, 0, 0.3)"
