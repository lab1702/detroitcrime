"""Data loading, cleaning and year-over-year maths."""

import gc
import logging
import threading
import weakref
from datetime import date

import pandas as pd
import plotly.express as px
import pytest
import requests

import utils
from constants import REQUIRED_COLUMNS
from theme import BLUE_ACCENT, CATEGORY_CARD_KEY, CUSTOM_CSS, RED_ACCENT, apply_dark_plotly
from utils import (
    DataNotReadyError, LoadFailedError, RefreshingCache, clean_and_filter_data, code_labels,
    compute_yoy_deltas, daily_incident_counts, day_labels, format_categorical_series, format_change,
    format_daily_rate, holiday_label, numeric_codes, part_year_note, partial_years, project_full_year,
    refuse_shrunken_refresh, safe_load_data_from_url, same_day_last_year, selection_period,
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

    def test_stray_late_row_is_not_taken_for_the_last_day(self):
        # 20 incidents a day, and 6 so far on 4 Mar, the feed's real last day.
        # One row mistyped as 9 Mar must not pass for the last day in its place.
        def day(number: int, incidents: int) -> list[str]:
            return [f"2024/03/{number:02d} 15:{minute:02d}:00+00" for minute in range(incidents)]

        df = clean(feed_rows(day(1, 20) + day(2, 20) + day(3, 20) + day(4, 6) + day(9, 1)))
        assert df["date"].value_counts().to_dict() == {
            date(2024, 3, 1): 20, date(2024, 3, 2): 20, date(2024, 3, 3): 20,
        }

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

    def test_timestamp_with_nanoseconds_does_not_fail_the_load(self):
        # Read on its own it comes back in a finer unit than the column holds
        good = [f"2024/03/{day:02d} 15:00:00+00" for day in range(1, 29)] * 8
        df = clean(feed_rows(good + ["2024-03-03T15:00:00.1234567Z", LATER]))
        assert len(df) == len(good) + 1
        assert (df["date"] == date(2024, 3, 3)).sum() == 9

    def test_padding_only_timestamp_is_missing_not_unreadable(self):
        # As unreadable ones, 2 in 52 would be over the limit and refuse the load
        good = [f"2024/03/{day:02d} 15:00:00+00" for day in range(1, 26)] * 2
        df = clean(feed_rows(good + ["      ", "", LATER]))
        assert len(df) == len(good)

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


class TestBadCoordinates:
    def test_non_numeric_coordinate_is_dropped_without_failing_the_load(self):
        # One bad cell makes read_csv hand over the whole column as text
        df = clean(feed_rows(
            ["2024/03/01 15:00:00+00", "2024/03/01 16:00:00+00", "2024/03/01 17:00:00+00", LATER],
            latitude=["42.33", "N/A", "42.35", "42.33"],
        ))
        assert df["latitude"].tolist() == [42.33, 42.35]


class TestDayAndHourNumbers:
    def test_cell_that_is_not_a_number_is_missing_and_the_rest_are_numbers(self):
        # One bad cell makes read_csv hand over the whole column as text
        df = clean(feed_rows(
            ["2024/03/01 15:00:00+00", "2024/03/01 16:00:00+00", "2024/03/01 17:00:00+00", LATER],
            incident_day_of_week=["5", "UNK", "5", "2"],
            incident_hour_of_day=["10", "11", "x", "7"],
        ))
        assert df["incident_day_of_week"].isna().tolist() == [False, True, False]
        assert df["incident_day_of_week"].dropna().tolist() == [5, 5]
        assert df["incident_hour_of_day"].isna().tolist() == [False, False, True]
        assert df["incident_hour_of_day"].dropna().tolist() == [10, 11]

    def test_one_empty_cell_does_not_turn_the_rest_into_floats(self):
        # With an empty cell read_csv hands the column over as floats: 5.0, 10.0
        df = clean(feed_rows(
            ["2024/03/01 15:00:00+00", "2024/03/01 16:00:00+00", "2024/03/01 17:00:00+00", LATER],
            incident_day_of_week=[5.0, None, 5.0, 2.0],
            incident_hour_of_day=[10.0, 11.0, 3.5, 7.0],
        ))
        assert pd.api.types.is_integer_dtype(df["incident_day_of_week"])
        assert pd.api.types.is_integer_dtype(df["incident_hour_of_day"])
        assert [str(day) for day in df["incident_day_of_week"].dropna()] == ["5", "5"]
        # Half past three is not an hour of the day
        assert df["incident_hour_of_day"].isna().tolist() == [False, False, True]
        assert [str(hour) for hour in df["incident_hour_of_day"].dropna()] == ["10", "11"]

    def test_numbers_are_left_as_they_are(self, raw_feed, clean_df):
        assert pd.api.types.is_integer_dtype(clean_df["incident_day_of_week"])
        assert set(clean_df["incident_hour_of_day"]) == set(range(24))


class TestTextCleanup:
    def test_padding_only_is_a_missing_value(self):
        df = clean(feed_rows(
            ["2024/03/01 15:00:00+00", "2024/03/01 16:00:00+00", LATER],
            neighborhood=["      ", "Nolan  ", "x"],
        ))
        assert df["neighborhood"].isna().tolist() == [True, False]
        assert df["neighborhood"].value_counts().to_dict() == {"Nolan": 1}

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
    # First day of a dataset that reaches back well before the years compared
    START = date(2018, 1, 1)

    def test_projection_uses_days_of_data(self):
        # 59 days of data at 10 a day projects to a full year at the same rate
        assert project_full_year(590, date(2025, 2, 28), self.START) is None  # under 60 days
        assert project_full_year(600, date(2025, 3, 1), self.START) == pytest.approx(10 * 365)
        assert project_full_year(600, date(2024, 2, 29), self.START) == pytest.approx(10 * 366)

    def test_projection_counts_only_the_days_the_data_covers(self):
        # The data starts on 1 February: 149 days at 8 a day by 29 June. Scaled
        # from 1 January, as 180 days, this would come to 2,417 for the year.
        assert project_full_year(1192, date(2026, 6, 29), date(2026, 2, 1)) == pytest.approx(8 * 365)
        # The 60 days needed are days of data, not days of the year
        assert project_full_year(480, date(2026, 6, 29), date(2026, 5, 1)) == pytest.approx(8 * 365)
        assert project_full_year(472, date(2026, 6, 29), date(2026, 5, 2)) is None

    def test_years_covered_in_part(self):
        # First and last year of a dataset that starts in June and ends in March
        assert partial_years(date(2024, 6, 1), date(2026, 3, 30)) == {
            2024: (date(2024, 6, 1), date(2024, 12, 31)),
            2026: (date(2026, 1, 1), date(2026, 3, 30)),
        }
        # Whole years are not listed, and a dataset within one year lists it once
        assert partial_years(date(2018, 1, 1), date(2026, 12, 31)) == {}
        assert partial_years(date(2018, 1, 1), date(2026, 2, 9)) == {2026: (date(2026, 1, 1), date(2026, 2, 9))}
        assert partial_years(date(2026, 2, 1), date(2026, 6, 29)) == {2026: (date(2026, 2, 1), date(2026, 6, 29))}

    def test_part_of_a_year_in_a_few_words(self):
        assert part_year_note(date(2026, 1, 1), date(2026, 2, 9)) == "to 9 Feb"
        assert part_year_note(date(2024, 6, 1), date(2024, 12, 31)) == "from 1 Jun"
        assert part_year_note(date(2026, 2, 1), date(2026, 6, 29)) == "1 Feb to 29 Jun"

    def test_change_that_rounds_to_zero_has_no_sign(self):
        assert format_change(0.042) == "+4.2%"
        assert format_change(-0.003) == "-0.3%"
        assert format_change(1.0) == "+100.0%"
        # "+0.0%" would be drawn as a rise and "-0.0%" as a fall
        assert format_change(0.0) == format_change(0.0004) == format_change(-0.0004) == "0.0%"
        assert format_change(0.0006) == "+0.1%"

    @staticmethod
    def yearly(prior_dates, current_dates) -> pd.DataFrame:
        dates = pd.to_datetime(list(prior_dates) + list(current_dates))
        return pd.DataFrame({"incident_year": dates.year, "date": dates.date, "incident_occurred_at": dates})

    def test_flat_year_shows_no_change_whatever_today_is(self):
        prior = pd.date_range("2025-01-01", "2025-12-31", freq="D")
        current = pd.date_range("2026-01-01", "2026-03-06", freq="D")  # data ends 6 Mar
        incidents, avg = compute_yoy_deltas(self.yearly(prior, current), date(2026, 3, 6), self.START)
        assert incidents == "0.0%"
        assert avg == "0.0%"

    def test_seasons_are_not_taken_for_a_change(self):
        # Two identical years, quiet from December to March. Measured against
        # the whole of last year, the first quarter would look like a fall of a third.
        def year(number: int, through: str) -> list:
            days = pd.date_range(f"{number}-01-01", through, freq="D")
            return [day for day in days for _ in range(3 if day.month in (12, 1, 2, 3) else 5)]

        frame = self.yearly(year(2025, "2025-12-31"), year(2026, "2026-03-31"))
        assert compute_yoy_deltas(frame, date(2026, 3, 31), self.START) == ("0.0%", "0.0%")

    def test_only_the_same_days_of_last_year_are_compared(self):
        # Last year: 10 incidents by the end of March, then 100 in November
        prior = ["2025-02-10"] * 10 + ["2025-11-05"] * 100
        current = ["2026-02-10"] * 20
        assert compute_yoy_deltas(self.yearly(prior, current), date(2026, 3, 31), self.START) == ("+100.0%", "+100.0%")

    def test_last_day_of_data_is_counted_in_both_years(self):
        # An incident late on the last day, and one on the same day last year
        prior = ["2025-03-31 23:30"] * 4 + ["2025-04-01 00:10"]
        current = ["2026-03-31 23:30"] * 5
        assert compute_yoy_deltas(self.yearly(prior, current), date(2026, 3, 31), self.START) == ("+25.0%", "+25.0%")

    def test_sparse_prior_year_deltas_agree(self):
        # 2 incidents on consecutive days last year, 20 by day 300 this year
        prior = ["2025-03-03", "2025-03-04"]
        current = pd.date_range("2026-01-01", periods=20, freq="15D")
        incidents, avg = compute_yoy_deltas(self.yearly(prior, current), date(2026, 10, 27), self.START)
        assert incidents == avg == "+900.0%"

    def test_leap_day_counts_in_the_total_but_not_in_the_daily_average(self):
        # One incident a day in both years; 2028 has a 29 February, 2027 does not
        prior = pd.date_range("2027-01-01", "2027-12-31", freq="D")
        current = pd.date_range("2028-01-01", "2028-02-29", freq="D")
        assert same_day_last_year(date(2028, 2, 29)) == date(2027, 2, 28)
        assert same_day_last_year(date(2028, 3, 1)) == date(2027, 3, 1)
        # 60 days against 59
        assert compute_yoy_deltas(self.yearly(prior, current), date(2028, 2, 29), self.START) == ("+1.7%", "0.0%")

    def test_no_delta_without_both_years(self):
        current = pd.date_range("2026-01-01", "2026-06-30", freq="D")
        assert compute_yoy_deltas(self.yearly([], current), date(2026, 6, 30), self.START) == (None, None)

    def test_no_delta_until_enough_of_the_year_has_data(self):
        prior = pd.date_range("2025-01-01", "2025-12-31", freq="D")
        current = pd.date_range("2026-01-01", "2026-02-28", freq="D")  # 59 days
        assert compute_yoy_deltas(self.yearly(prior, current), date(2026, 2, 28), self.START) == (None, None)

    def test_no_delta_unless_the_data_covers_last_year_from_its_first_day(self):
        # One incident a day from 15 February last year. Its 44 days up to 30
        # March would be set against this year's 89, and show as a doubling.
        prior = pd.date_range("2025-02-15", "2025-12-31", freq="D")
        current = pd.date_range("2026-01-01", "2026-03-30", freq="D")
        frame = self.yearly(prior, current)
        assert compute_yoy_deltas(frame, date(2026, 3, 30), date(2025, 2, 15)) == (None, None)
        assert compute_yoy_deltas(frame, date(2026, 3, 30), date(2025, 1, 2)) == (None, None)
        # A dataset that starts on 1 January of last year, or before, is compared
        whole_prior = pd.date_range("2025-01-01", "2025-12-31", freq="D")
        frame = self.yearly(whole_prior, current)
        assert compute_yoy_deltas(frame, date(2026, 3, 30), date(2025, 1, 1)) == ("0.0%", "0.0%")


class TestCategoricalFormatting:
    def test_series_drops_invalid_and_adds_prefix(self):
        codes = pd.Series([3.0, 11, "07", None, 0, "n/a"], index=list("abcdef"))
        assert format_categorical_series(codes, "Precinct").to_dict() == {
            "a": "Precinct 3", "b": "Precinct 11", "c": "Precinct 7",
        }
        assert format_categorical_series(pd.Series([48226.0, None])).tolist() == ["48226"]

    def test_codes_are_numbers_with_zero_for_anything_invalid(self):
        assert numeric_codes(pd.Series([3.0, "07", None, "n/a", 0, "48201-1234"])).tolist() == [3, 7, 0, 0, 0, 0]

    def test_only_whole_finite_numbers_are_codes(self):
        # An infinity used to fail the whole column; 3.7 was counted as code 3
        codes = pd.Series(["inf", "1e400", 3.7, "4.0", 1e300])
        assert numeric_codes(codes).tolist() == [0, 0, 0, 4, 0]
        assert numeric_codes(pd.Series([2, 12])).tolist() == [2, 12]

    def test_codes_are_above_zero_and_written_as_plain_numbers(self):
        assert numeric_codes(pd.Series([-3, 5, 0])).tolist() == [0, 5, 0]
        assert format_categorical_series(pd.Series([-3, 5]), "Precinct").tolist() == ["Precinct 5"]
        # A parser takes these for 1000, 16 and 1; nobody writes a precinct that way
        odd = pd.Series(["1e3", "0x10", "True", " 7 ", "-2", True])
        assert numeric_codes(odd).tolist() == [0, 0, 0, 7, 0, 0]
        assert numeric_codes(pd.Series([True, False])).tolist() == [0, 0]

    def test_labels_keep_the_order_of_the_codes(self):
        assert code_labels([2, 9, 10], "Precinct") == ["Precinct 2", "Precinct 9", "Precinct 10"]
        assert code_labels([48201, 48226]) == ["48201", "48226"]

    def test_days_are_named_iso_style(self):
        assert day_labels([1, 2.0, 7]) == ["Monday", "Tuesday", "Sunday"]
        assert day_labels([0, 8]) == ["0", "8"]

    def test_holiday_names_fit_beside_their_marks(self):
        assert holiday_label("Labor Day") == "Labor Day"
        assert holiday_label("Independence Day") == "Independence Day"
        # Cut to nine characters, these two used to read "Juneteent..." and "Martin Lu..."
        assert holiday_label("Juneteenth National Independence Day") == "Juneteenth"
        assert holiday_label("Martin Luther King Jr. Day") == "MLK Day"
        assert holiday_label("Washington's Birthday") == "Presidents' Day"

    def test_unfamiliar_long_holiday_name_is_cut_at_a_word(self):
        assert holiday_label("Some Very Long Holiday Name") == "Some Very Long…"
        assert holiday_label("Supercalifragilisticexpialidocious Day") == "Supercalifragil…"
        assert all(len(holiday_label(name)) <= 16 for name in ("A Day Exactly Sixteen", "Sixteen Letters!!", "Day"))


class TestDailyRate:
    def test_one_decimal_from_one_a_day_up(self):
        assert format_daily_rate(20.04) == "20.0"
        assert format_daily_rate(1.0) == "1.0"
        assert format_daily_rate(1.04) == "1.0"

    def test_two_significant_digits_below_one_a_day(self):
        assert format_daily_rate(0.95) == "0.95"
        assert format_daily_rate(0.5) == "0.5"
        assert format_daily_rate(3 / 668) == "0.0045"

    def test_rate_that_rounds_up_to_one_reads_like_one(self):
        # ".2g" gave a bare "1" here, next to "1.0" for a rate just above
        assert format_daily_rate(0.996) == "1.0"
        assert format_daily_rate(0.9999) == "1.0"

    def test_very_small_rate_is_written_out(self):
        assert format_daily_rate(0.00009) == "0.00009"  # not "9e-05"
        assert format_daily_rate(0.0) == "0"


class TestValidation:
    def test_missing_columns_are_named_even_when_none_are_there(self):
        # A frame loaded with none of its columns is empty too; what it lacks is the news
        with pytest.raises(ValueError, match="Missing required columns"):
            utils.validate_data(pd.DataFrame(index=range(3)))

    def test_no_rows_is_an_error(self):
        with pytest.raises(ValueError, match="empty"):
            utils.validate_data(pd.DataFrame(columns=REQUIRED_COLUMNS))
        with pytest.raises(ValueError, match="empty"):
            utils.validate_data(None)


class TestTheme:
    def test_category_card_is_styled_by_the_key_the_page_gives_it(self):
        # The page and the stylesheet each name the card; they have to agree
        assert f".st-key-{CATEGORY_CARD_KEY} " in CUSTOM_CSS


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
        # A quoted value that never ends: the same file would fail the same way
        portal.extend([FakeResponse(b'latitude,offense_category\n42.3,"ARSON\n'), FakeResponse()])
        with pytest.raises(pd.errors.ParserError):
            safe_load_data_from_url(URL)
        assert len(portal) == 1

    @pytest.mark.parametrize("body", [
        b"",
        b"\n\n",
        b"latitude,offense_category\n",
        b"X,Y,ESRI_OID\n",
        b"The dataset is being updated\n",
    ], ids=["no bytes", "blank lines", "header only", "header only, other columns", "one line of text"])
    def test_file_with_no_rows_is_not_ready_and_is_tried_again(self, portal, body):
        # The portal empties the dataset while it reloads it; an export made
        # then has no rows. That passes, unlike a feed that has changed.
        portal.extend([FakeResponse(body), FakeResponse(body)])
        with pytest.raises(DataNotReadyError, match="empty file|no rows"):
            safe_load_data_from_url(URL, max_retries=2, usecols=REQUIRED_COLUMNS)
        assert portal == []
        assert utils.is_transient_error(DataNotReadyError("the portal sent a file with no rows in it"))

    def test_file_with_rows_and_none_of_the_columns_is_left_to_validation(self, portal):
        # Rows, but under other names: the feed has changed, and waiting will
        # not change it back. It is loaded once and reported by column.
        portal.extend([FakeResponse(b"X,Y,ESRI_OID\n1,2,3\n4,5,6\n"), FakeResponse()])
        df = safe_load_data_from_url(URL, usecols=REQUIRED_COLUMNS)
        assert len(portal) == 1
        with pytest.raises(ValueError, match="Missing required columns: .*'latitude'.*'zip_code'"):
            utils.validate_data(df)

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

    def test_stray_invalid_byte_costs_one_character_not_the_load(self, portal):
        portal.append(FakeResponse(b"latitude,nearest_intersection\n42.3,Caf\xe9 Ave\n"))
        df = safe_load_data_from_url(URL)
        assert df["nearest_intersection"].tolist() == ["Caf� Ave"]

    def test_makes_no_streamlit_calls(self, portal, monkeypatch):
        # It runs on the cache's thread, which has no page to draw on. On a
        # session's thread a Streamlit call is where a click interrupts the
        # script, which used to throw a finished download away.
        class NoStreamlit:
            def __getattr__(self, name):
                raise AssertionError(f"st.{name} was used during the download")

        monkeypatch.setattr(utils, "st", NoStreamlit())
        portal.extend([FakeResponse(status=503), FakeResponse()])
        assert len(safe_load_data_from_url(URL)) == 2


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
        return RefreshingCache(ttl=100, retry_interval=10, first_retry_interval=3, clock=clock)

    def test_value_is_reused_until_it_is_due(self, cache, clock):
        load = loader("first", "second")
        assert cache.get(load) == ("first", None)
        clock.now += 99
        assert cache.get(load) == ("first", None)
        clock.now += 1
        # Due: the refresh runs behind the old value, which is what this call gets
        assert cache.get(load) == ("first", None)
        assert cache.wait(timeout=5)
        assert cache.get(load) == ("second", None)

    def test_refresh_keeps_nobody_waiting(self, cache, clock):
        cache.get(loader("first"))
        clock.now += 100
        release = threading.Event()

        def slow_refresh():
            release.wait(timeout=5)
            return "second"

        # Neither the caller that sets the refresh off nor one that comes during it waits
        assert cache.get(slow_refresh) == ("first", None)
        assert cache.get(loader()) == ("first", None)
        assert not cache.wait(timeout=0)  # still running
        release.set()
        assert cache.wait(timeout=5)
        assert cache.get(loader()) == ("second", None)

    def test_failed_refresh_keeps_the_previous_value(self, cache, clock):
        down = RuntimeError("portal down")
        load = loader("first", down, "second")
        cache.get(load)

        clock.now += 100
        assert cache.get(load) == ("first", None)
        assert cache.wait(timeout=5)
        assert cache.get(load) == ("first", down)
        # Not tried again until the retry interval has passed
        clock.now += 9
        assert cache.get(load) == ("first", down)
        clock.now += 1
        # While it is tried again the last failure still stands
        assert cache.get(load) == ("first", down)
        assert cache.wait(timeout=5)
        assert cache.get(load) == ("second", None)

    def test_first_load_failure_is_raised_and_nothing_is_kept(self, cache, clock):
        down = RuntimeError("portal down")
        load = loader(down, "first")
        with pytest.raises(LoadFailedError, match="portal down") as first:
            cache.get(load)
        assert first.value.__cause__ is down
        # The next caller is told of the same failure; the load is not run again for it...
        clock.now += 2
        with pytest.raises(LoadFailedError, match="portal down") as second:
            cache.get(load)
        assert second.value.__cause__ is down
        # ...until the wait before another first load is over
        clock.now += 1
        assert cache.get(load) == ("first", None)

    def test_each_caller_gets_an_error_of_its_own(self, cache):
        # The failure is shared by every caller and thread, so it is never raised
        # again: each raise would rewrite its traceback under the others
        down = RuntimeError("portal down")
        load = loader(down)
        raised = []
        for _ in range(2):
            with pytest.raises(LoadFailedError) as caught:
                cache.get(load)
            raised.append(caught.value)

        assert raised[0] is not raised[1]
        assert raised[0].__cause__ is raised[1].__cause__ is down
        assert down.__traceback__ is None

    def test_load_that_ends_badly_still_lets_go_of_everyone(self, cache, clock, monkeypatch):
        # Tidying up after a failed load fails as well, as it can once memory has run out
        def out_of_memory(error):
            raise MemoryError("nothing left")

        monkeypatch.setattr(utils, "_release_frames", out_of_memory)
        monkeypatch.setattr(threading, "excepthook", lambda args: None)  # the load thread's last words

        seen = []

        def session():
            try:
                cache.get(loader(RuntimeError("portal down")))
            except LoadFailedError as error:
                seen.append(str(error))

        waiting = threading.Thread(target=session, daemon=True)
        waiting.start()
        waiting.join(timeout=5)
        assert seen == ["portal down"]

        # The cache does not go on believing that load is under way
        clock.now += 3
        assert cache.get(loader("first")) == ("first", None)

    def test_callers_waiting_for_the_first_load_share_it(self, cache):
        started, finish = threading.Event(), threading.Event()
        loads = []

        def slow_load():
            loads.append("load")
            started.set()
            finish.wait(timeout=5)
            return "first"

        seen = []
        sessions = [threading.Thread(target=lambda: seen.append(cache.get(slow_load))) for _ in range(4)]
        for session in sessions:
            session.start()
        started.wait(timeout=5)
        finish.set()
        for session in sessions:
            session.join(timeout=5)

        assert loads == ["load"]
        assert seen == [("first", None)] * 4

    def test_failed_first_load_fails_once_for_everyone_waiting(self, cache):
        started, finish = threading.Event(), threading.Event()
        loads = []

        def slow_failure():
            loads.append("load")
            started.set()
            finish.wait(timeout=5)
            raise RuntimeError("portal down")

        seen = []

        def session():
            try:
                cache.get(slow_failure)
            except LoadFailedError as error:
                seen.append(str(error))

        sessions = [threading.Thread(target=session) for _ in range(4)]
        for thread in sessions:
            thread.start()
        started.wait(timeout=5)
        finish.set()
        for thread in sessions:
            thread.join(timeout=5)

        assert loads == ["load"]
        assert seen == ["portal down"] * 4

    def test_load_does_not_run_on_the_thread_that_asked_for_it(self, cache):
        # A session's script can be interrupted while it waits (its user clicked
        # something). The load is not on that thread, so it finishes and is kept.
        ran_on = []
        assert cache.get(lambda: ran_on.append(threading.current_thread()) or "first") == ("first", None)
        assert ran_on[0] is not threading.current_thread()

    def test_retry_interval_starts_when_the_failed_attempt_ends(self, cache, clock):
        cache.get(loader("first"))
        clock.now += 100

        def slow_failure():
            clock.now += 60  # the attempt alone outlasts the retry interval
            raise RuntimeError("portal down")

        cache.get(slow_failure)
        assert cache.wait(timeout=5)
        # Straight afterwards the portal is left alone (this loader would fail if run)
        value, error = cache.get(loader())
        assert (value, str(error)) == ("first", "portal down")
        clock.now += 9
        assert cache.get(loader()) == ("first", error)
        clock.now += 1
        cache.get(loader("second"))
        assert cache.wait(timeout=5)
        assert cache.get(loader()) == ("second", None)

    def test_kept_failure_does_not_keep_the_failed_load_in_memory(self, cache):
        class Download:
            """Stands in for the dataset a failed load was holding."""

        downloads = []

        def failing_load():
            download = Download()
            downloads.append(weakref.ref(download))
            try:
                raise KeyError("zip_code")
            except KeyError:
                raise ValueError("the feed changed")  # the KeyError is kept as its context

        cache.get(loader("first"))
        cache.expire()
        cache.get(failing_load)
        assert cache.wait(timeout=5)
        value, error = cache.get(loader())
        gc.collect()
        assert (value, str(error)) == ("first", "the feed changed")
        assert downloads[0]() is None

    def test_failed_load_is_logged_with_where_it_went_wrong(self, cache, caplog):
        # The pages show the message only, and the traceback is not kept
        def failing_load():
            raise ValueError("the feed changed")

        with caplog.at_level(logging.ERROR, logger="utils"):
            with pytest.raises(LoadFailedError):
                cache.get(failing_load)

        [record] = [r for r in caplog.records if r.name == "utils"]
        assert record.levelname == "ERROR"
        assert "ValueError: the feed changed" in caplog.text
        assert "in failing_load" in caplog.text

    def test_refresh_that_fails_its_check_is_refused(self, clock):
        compared = []

        def check(new, previous):
            compared.append((new, previous))
            if new == "half of it":
                raise ValueError("too little")

        cache = RefreshingCache(ttl=100, retry_interval=10, clock=clock, check=check)
        load = loader("first", "half of it", "second")
        # A first load has nothing to be compared with
        assert cache.get(load) == ("first", None)
        assert compared == []

        # The refused refresh is a failed one: the value in use stays, with the reason
        clock.now += 100
        cache.get(load)
        assert cache.wait(timeout=5)
        value, error = cache.get(load)
        assert (value, str(error)) == ("first", "too little")
        assert compared == [("half of it", "first")]

        # The next refresh is checked against the value in use, and taken
        clock.now += 10
        cache.get(load)
        assert cache.wait(timeout=5)
        assert cache.get(load) == ("second", None)
        assert compared[-1] == ("second", "first")

    def test_refused_refresh_is_not_kept_in_memory(self, clock):
        class Download:
            """Stands in for a dataset."""

        def check(new, previous):
            raise ValueError("too little")

        cache = RefreshingCache(ttl=100, retry_interval=10, clock=clock, check=check)
        refused = []

        def refresh():
            download = Download()
            refused.append(weakref.ref(download))
            return download

        cache.get(loader("first"))
        cache.expire()
        cache.get(refresh)
        assert cache.wait(timeout=5)
        value, error = cache.get(loader())
        gc.collect()
        assert (value, str(error)) == ("first", "too little")
        assert refused[0]() is None

    def test_expire_makes_the_next_get_refresh(self, cache):
        load = loader("first", "second")
        cache.get(load)
        cache.expire()
        assert cache.get(load) == ("first", None)
        assert cache.wait(timeout=5)
        assert cache.get(load) == ("second", None)

    @staticmethod
    def attempts_over(cache: RefreshingCache, clock: Clock, seconds: int, load, step: int = 5) -> list[float]:
        """When, from now, `load` is run if the cache is asked every `step` seconds."""
        started, ran = clock.now, []

        def counted():
            ran.append(clock.now - started)
            return load()

        for _ in range(seconds // step):
            try:
                cache.get(counted)
            except LoadFailedError:
                pass
            assert cache.wait(timeout=5)
            clock.now += step
        return ran

    def test_refresh_that_would_only_repeat_is_tried_less_and_less_often(self, clock):
        # A feed that has lost a column fails the same way on every try, and
        # each try is a whole download: every 10 seconds, it was 100 of them here
        def changed_feed():
            raise ValueError("the feed changed")

        cache = RefreshingCache(
            ttl=100, retry_interval=10, clock=clock, can_pass=lambda error: not isinstance(error, ValueError),
        )
        cache.get(loader("first"))
        clock.now += 100
        # The wait doubles each time, and stops growing at the time to live
        assert self.attempts_over(cache, clock, 1000, changed_feed) == [
            0, 10, 30, 70, 150, 250, 350, 450, 550, 650, 750, 850, 950,
        ]
        value, error = cache.get(loader())
        assert (value, str(error)) == ("first", "the feed changed")

    def test_wait_for_a_repeating_failure_stops_growing_at_its_limit(self, clock):
        def changed_feed():
            raise ValueError("the feed changed")

        cache = RefreshingCache(
            ttl=1000, retry_interval=10, clock=clock, can_pass=lambda error: False, max_retry_interval=40,
        )
        cache.get(loader("first"))
        clock.now += 1000
        # 10, 20, then 40 at a time
        assert self.attempts_over(cache, clock, 200, changed_feed) == [0, 10, 30, 70, 110, 150, 190]

    def test_failure_that_can_pass_is_tried_again_at_the_usual_interval(self, clock):
        def portal_down():
            raise ConnectionError("portal down")

        cache = RefreshingCache(
            ttl=100, retry_interval=10, clock=clock, can_pass=lambda error: not isinstance(error, ValueError),
        )
        cache.get(loader("first"))
        clock.now += 100
        assert self.attempts_over(cache, clock, 50, portal_down) == [0, 10, 20, 30, 40]

    def test_wait_starts_over_once_a_failure_is_not_the_repeating_kind(self, clock):
        failures = [ValueError("changed"), ValueError("changed"), ConnectionError("down"), ValueError("changed")]

        def fail():
            raise failures.pop(0) if failures else ValueError("changed")

        cache = RefreshingCache(
            ttl=1000, retry_interval=10, clock=clock, can_pass=lambda error: not isinstance(error, ValueError),
        )
        cache.get(loader("first"))
        clock.now += 1000
        # 10 after the first, 20 after the second, then the usual 10 after the
        # fault on the line and after the first repeating failure that follows it
        assert self.attempts_over(cache, clock, 80, fail) == [0, 10, 30, 40, 50, 70]

    def test_repeating_first_load_failure_backs_off_from_the_first_load_interval(self, clock):
        def changed_feed():
            raise ValueError("the feed changed")

        cache = RefreshingCache(
            ttl=100, retry_interval=50, first_retry_interval=5, clock=clock, can_pass=lambda error: False,
        )
        assert self.attempts_over(cache, clock, 100, changed_feed) == [0, 5, 15, 35, 75]

    def test_a_good_load_ends_the_long_wait(self, clock):
        results = [ValueError("changed"), ValueError("changed"), "second", ValueError("changed"), ValueError("changed")]

        def load():
            result = results.pop(0)
            if isinstance(result, Exception):
                raise result
            return result

        cache = RefreshingCache(ttl=100, retry_interval=10, clock=clock, can_pass=lambda error: False)
        cache.get(loader("first"))
        clock.now += 100
        # Fails at 0 and 10, loads at 30; due again at 130, and the failure
        # then is left alone for the usual 10, not for twice the last wait
        assert self.attempts_over(cache, clock, 150, load) == [0, 10, 30, 130, 140]

    def test_up_to_date_until_a_load_starts_or_ends(self, cache, clock):
        value, error = cache.get(loader("first"))
        assert cache.up_to_date(value, error)

        clock.now += 100
        release = threading.Event()

        def slow_refresh():
            release.wait(timeout=5)
            return "second"

        value, error = cache.get(slow_refresh)
        assert not cache.up_to_date(value, error)  # a refresh is under way
        release.set()
        assert cache.wait(timeout=5)
        # Nothing is running now, but what the caller was given has been replaced.
        # A caller that only asked whether a load was running would never know.
        assert cache.wait(timeout=0)
        assert not cache.up_to_date(value, error)
        assert cache.up_to_date(*cache.get(loader()))

    def test_not_up_to_date_once_a_refresh_has_failed(self, cache, clock):
        value, error = cache.get(loader("first"))
        clock.now += 100
        assert cache.get(loader(RuntimeError("portal down"))) == (value, None)
        assert cache.wait(timeout=5)
        # The value is the same one, but there is a failure to report with it now
        assert not cache.up_to_date(value, None)
        assert cache.up_to_date(*cache.get(loader()))


class TestShrunkenRefresh:
    @staticmethod
    def rows(count: int) -> pd.DataFrame:
        return pd.DataFrame({"latitude": [42.33] * count})

    def test_refresh_that_grows_or_loses_a_little_is_taken(self):
        refuse_shrunken_refresh(self.rows(1010), self.rows(1000))
        refuse_shrunken_refresh(self.rows(1000), self.rows(1000))
        refuse_shrunken_refresh(self.rows(900), self.rows(1000))

    def test_refresh_that_has_lost_much_of_its_rows_is_refused(self):
        with pytest.raises(ValueError, match="only 899 usable incidents, against 1,000 in the copy shown"):
            refuse_shrunken_refresh(self.rows(899), self.rows(1000))

    def test_dataset_cache_checks_its_refreshes(self):
        utils.st.cache_resource.clear()
        try:
            cache = utils.dataset_cache()
            assert len(cache.get(lambda: self.rows(1000))[0]) == 1000
            cache.expire()
            cache.get(lambda: self.rows(100))
            assert cache.wait(timeout=5)
            value, error = cache.get(lambda: self.rows(100))
            assert len(value) == 1000
            assert "only 100 usable incidents" in str(error)
            # A refused refresh would be refused again: it is one of the
            # failures the cache tries again less and less often
            assert cache._can_pass is utils.is_transient_error
            assert not utils.is_transient_error(error)
            assert cache._max_retry_interval == utils.DATA_MAX_RETRY_INTERVAL < utils.DATA_CACHE_TTL
        finally:
            utils.st.cache_resource.clear()


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
