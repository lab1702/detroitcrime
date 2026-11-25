"""Data loading, cleaning and year-over-year maths."""

from datetime import date

import pandas as pd
import plotly.express as px
import pytest

import utils
from constants import REQUIRED_COLUMNS
from theme import BLUE_ACCENT, RED_ACCENT, apply_dark_plotly
from utils import (
    clean_and_filter_data, compute_yoy_deltas, format_categorical_column,
    format_categorical_series, project_full_year, safe_load_data_from_url,
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


def clean(df: pd.DataFrame) -> pd.DataFrame:
    return clean_and_filter_data(df, "2018-01-01", *BOUNDS)


# A later row so the incidents under test are not on the dropped "most recent" day
LATER = "2030/01/01 12:00:00+00"


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

    def test_column_version_matches_series_version(self):
        df = pd.DataFrame({"police_precinct": [3.0, None, 12], "n": [1, 2, 3]})
        out = format_categorical_column(df, "police_precinct", "Precinct")
        assert out["police_precinct"].tolist() == ["Precinct 3", "Precinct 12"]
        assert out["n"].tolist() == [1, 3]
        assert df["police_precinct"].tolist()[0] == 3.0  # input untouched


class TestDownload:
    class FakeResponse:
        def __init__(self, body: bytes):
            self.body = body

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def raise_for_status(self):
            pass

        def iter_content(self, chunk_size):
            for i in range(0, len(self.body), 7):  # small, uneven chunks
                yield self.body[i:i + 7]

    def test_reads_only_wanted_columns_from_a_bom_csv(self, monkeypatch):
        body = "﻿X,latitude,offense_category,ESRI_OID\n1,42.3,ARSON,9\n2,42.4,FRAUD,8\n".encode("utf-8")
        monkeypatch.setattr(utils.requests, "get", lambda *a, **k: self.FakeResponse(body))

        df = safe_load_data_from_url("https://example.invalid/feed.csv", usecols=REQUIRED_COLUMNS)

        # Columns the feed lacks are left for validate_data to report
        assert df.columns.tolist() == ["latitude", "offense_category"]
        assert df["offense_category"].tolist() == ["ARSON", "FRAUD"]

    def test_loads_everything_without_usecols(self, monkeypatch):
        body = b"a,b\n1,2\n"
        monkeypatch.setattr(utils.requests, "get", lambda *a, **k: self.FakeResponse(body))
        assert safe_load_data_from_url("https://example.invalid/feed.csv").columns.tolist() == ["a", "b"]


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
