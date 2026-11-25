"""Utility functions for Detroit Crime Dashboard

This module provides common utility functions for data loading, validation,
and processing used throughout the crime dashboard application.
"""

import calendar
import tempfile
import threading
import time
from datetime import date
from typing import Any, Callable, Iterable, Optional, Tuple

import pandas as pd
import requests
import streamlit as st

from constants import (
    DATA_CACHE_TTL, DATA_RETRY_INTERVAL, LOCAL_TIMEZONE, MAX_UNPARSEABLE_DATE_SHARE,
    MIN_PROJECTION_DAYS, REQUIRED_COLUMNS, TEXT_COLUMNS,
)


def validate_data(df: Optional[pd.DataFrame]) -> bool:
    """Validate that the dataframe has required columns and basic data quality"""
    if df is None or df.empty:
        raise ValueError("DataFrame is empty or None")

    # Check for required columns
    missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Check for basic data quality
    if df['latitude'].isna().all() or df['longitude'].isna().all():
        raise ValueError("All latitude/longitude values are missing")

    if df['incident_occurred_at'].isna().all():
        raise ValueError("All incident dates are missing")

    return True


def numeric_codes(series: pd.Series) -> pd.Series:
    """Parse a column of numeric codes to ints, with 0 for anything invalid."""
    return pd.to_numeric(series, errors='coerce').fillna(0).astype(int)


def code_labels(codes: Iterable[int], prefix: str = "") -> list[str]:
    """Labels for numeric codes: 3 becomes "Precinct 3", or "3" with no prefix."""
    return [f"{prefix} {code}" if prefix else str(code) for code in codes]


def format_categorical_series(series: pd.Series, prefix: str = "") -> pd.Series:
    """
    Format a column of numeric codes (precinct, district, zip) as labels.

    Invalid and zero entries are dropped, decimals are removed, and the
    optional prefix is added (e.g. "Precinct 3"). Works on a single column so
    callers do not need to copy the whole DataFrame.

    Args:
        series: Column to format
        prefix: Optional prefix to add (e.g., "Precinct", "District")

    Returns:
        Series of string labels for the valid entries only
    """
    codes = numeric_codes(series)
    labels = codes[codes != 0].astype(str)
    return f"{prefix} " + labels if prefix else labels


# ── Download ─────────────────────────────────────────────────────────────────

# Responses worth another attempt: rate limiting and trouble on the server
RETRY_STATUS_CODES = frozenset({429, 500, 502, 503, 504})

# Network failures that can clear on their own. ConnectionError covers refused
# and reset connections and read timeouts mid-download; ChunkedEncodingError is
# a download cut short (IncompleteRead).
TRANSIENT_ERRORS = (
    requests.exceptions.ConnectionError,
    requests.exceptions.Timeout,
    requests.exceptions.ChunkedEncodingError,
    requests.exceptions.ContentDecodingError,
)


class DataNotReadyError(Exception):
    """The portal answered, but not with the CSV (e.g. while it rebuilds the export)."""


def is_transient_error(error: Exception) -> bool:
    """Whether a failed download is worth another attempt.

    Network trouble, rate limiting, server errors and a portal that is not
    ready are; a client error (such as 404) or a file that cannot be parsed
    will fail the same way every time.
    """
    if isinstance(error, requests.exceptions.HTTPError):
        return getattr(error.response, "status_code", None) in RETRY_STATUS_CODES
    return isinstance(error, TRANSIENT_ERRORS + (DataNotReadyError,))


def _require_csv(start: bytes, content_type: str) -> None:
    """Raise DataNotReadyError when a response body is JSON or HTML, not CSV."""
    first = start.lstrip(b"\xef\xbb\xbf \t\r\n")[:1]
    if first in (b"{", b"[", b"<"):
        raise DataNotReadyError(
            f"the portal sent {content_type or 'an unlabelled response'} instead of the CSV"
        )


def safe_load_data_from_url(url: str, timeout: int = 300, max_retries: int = 5,
                            usecols: Optional[Iterable[str]] = None) -> pd.DataFrame:
    """
    Safely load data from URL with application-level retries.

    Retries the entire download on transient failures (see is_transient_error),
    including IncompleteRead errors that occur mid-stream during large
    downloads. Failures that would only repeat, such as a 404 or a file that
    cannot be parsed, are raised straight away.

    The download is spooled to a temporary file and parsed from there, so the
    only large object held in memory is the resulting DataFrame.

    Args:
        url: URL to fetch data from
        timeout: Read timeout in seconds
        max_retries: Number of full download attempts (at least 1)
        usecols: Columns to load; others are skipped. Columns missing from the
            feed are ignored here so validation can report them by name.

    Returns:
        DataFrame

    Raises:
        DataNotReadyError: the portal answered without the CSV
        requests.exceptions.RequestException: the download failed
        ValueError: the file could not be parsed
    """
    if max_retries < 1:
        raise ValueError("max_retries must be at least 1")

    wanted = set(usecols) if usecols is not None else None
    column_filter = (lambda name: name in wanted) if wanted is not None else None

    status_placeholder = st.empty()
    attempt = 0
    while True:
        attempt += 1
        try:
            with tempfile.TemporaryFile() as buffer:
                with st.spinner(f"Downloading Detroit crime data (attempt {attempt}/{max_retries})..."):
                    with requests.get(url, timeout=(15, timeout), stream=True) as response:
                        response.raise_for_status()
                        # e.g. 202 while the portal is still building the export
                        if response.status_code != 200:
                            raise DataNotReadyError(
                                f"the portal answered with status {response.status_code} instead of the data"
                            )
                        checked = False
                        for chunk in response.iter_content(chunk_size=1024 * 1024):
                            if not chunk:
                                continue
                            if not checked:
                                _require_csv(chunk, response.headers.get("Content-Type", ""))
                                checked = True
                            buffer.write(chunk)

                buffer.seek(0)
                df = pd.read_csv(buffer, encoding="utf-8-sig", usecols=column_filter, low_memory=False)

            status_placeholder.empty()
            return df

        except Exception as e:
            if attempt >= max_retries or not is_transient_error(e):
                status_placeholder.empty()
                raise
            wait = 2 ** attempt
            status_placeholder.warning(f"Download interrupted, retrying in {wait}s... ({e})")
            time.sleep(wait)


class RefreshingCache:
    """Keeps the last good result of an expensive load and refreshes it when stale.

    Unlike a cache with a time-to-live, the old value is not thrown away when
    it expires: it stays in use until a refresh succeeds. If a refresh fails,
    callers get the old value together with the error, and the load is not
    tried again until `retry_interval` has passed.
    """

    def __init__(self, ttl: float, retry_interval: float,
                 clock: Callable[[], float] = time.monotonic) -> None:
        self._ttl = ttl
        self._retry_interval = retry_interval
        self._clock = clock
        self._lock = threading.Lock()
        self._has_value = False
        self._value: Any = None
        self._loaded_at = 0.0
        self._failed_at: Optional[float] = None
        self._error: Optional[Exception] = None

    def get(self, loader: Callable[[], Any]) -> Tuple[Any, Optional[Exception]]:
        """Return (value, error), loading or refreshing with `loader` when due.

        `error` is None when the value is current. When a refresh has failed it
        is that failure, and the value is the previous one. With no previous
        value to fall back on, a failed load raises.
        """
        if self._has_value:
            # Someone else is refreshing: use what there is rather than wait
            if not self._lock.acquire(blocking=False):
                return self._value, self._error
        else:
            self._lock.acquire()

        try:
            now = self._clock()
            if self._has_value:
                if now - self._loaded_at < self._ttl:
                    return self._value, None
                if self._failed_at is not None and now - self._failed_at < self._retry_interval:
                    return self._value, self._error

            try:
                value = loader()
            except Exception as error:
                if not self._has_value:
                    raise
                self._failed_at, self._error = now, error
                return self._value, error

            self._value, self._has_value = value, True
            self._loaded_at, self._failed_at, self._error = now, None, None
            return value, None
        finally:
            self._lock.release()

    def expire(self) -> None:
        """Make the next `get` refresh the value, whatever its age."""
        with self._lock:
            self._loaded_at = self._clock() - self._ttl
            self._failed_at = None


@st.cache_resource(show_spinner=False)
def dataset_cache() -> RefreshingCache:
    """The one dataset cache shared by every session in this process.

    The DataFrame it holds is shared too. Treat it as read-only: filter or
    copy it before changing anything.
    """
    return RefreshingCache(ttl=DATA_CACHE_TTL, retry_interval=DATA_RETRY_INTERVAL)


# ── Cleaning ─────────────────────────────────────────────────────────────────

def parse_feed_timestamps(raw: pd.Series) -> pd.Series:
    """
    Parse the feed's UTC timestamps, tolerating a few bad values.

    Values that cannot be read become NaT, and are dropped later along with
    the rows that have no date at all. If more than MAX_UNPARSEABLE_DATE_SHARE
    of them cannot be read the feed itself has changed, and dropping that many
    incidents would understate every count, so that raises instead.

    Raises:
        ValueError: too many timestamps could not be read
    """
    try:
        occurred = pd.to_datetime(raw, utc=True, errors='coerce')
        unread = occurred.isna() & raw.notna()
        if unread.any():
            # Not written like the rest of the column: read those one by one
            occurred[unread] = pd.to_datetime(raw[unread], utc=True, errors='coerce', format='mixed')
            unread = occurred.isna() & raw.notna()
    except (ValueError, TypeError) as e:
        raise ValueError(f"Failed to parse incident dates: {str(e)}")

    present = int(raw.notna().sum())
    unreadable = int(unread.sum())
    if present and unreadable / present > MAX_UNPARSEABLE_DATE_SHARE:
        raise ValueError(
            f"Failed to parse incident dates: {unreadable:,} of {present:,} timestamps could not be read"
        )
    return occurred


def clean_and_filter_data(df: pd.DataFrame, min_date: str,
                         lat_bounds: Tuple[float, float],
                         lon_bounds: Tuple[float, float],
                         local_timezone: str = LOCAL_TIMEZONE,
                         today: Optional[date] = None) -> pd.DataFrame:
    """
    Clean and filter the crime data

    The feed publishes `incident_occurred_at` in UTC. It is converted to local
    wall-clock time (timezone-naive), and `date` and `incident_year` are derived
    from that, so an incident late in the evening is counted on the day and in
    the year it actually happened.

    Args:
        df: Raw DataFrame
        min_date: Minimum local date to include
        lat_bounds: Tuple of (min_lat, max_lat)
        lon_bounds: Tuple of (min_lon, max_lon)
        local_timezone: IANA timezone the incidents occurred in
        today: Current local date; incidents dated after it are dropped.
            Defaults to the current date in `local_timezone`.

    Returns:
        Cleaned DataFrame
    """
    # Clean and prepare data
    df = df.dropna(subset=['latitude', 'longitude'])

    occurred = parse_feed_timestamps(df['incident_occurred_at'])
    df['incident_occurred_at'] = occurred.dt.tz_convert(local_timezone).dt.tz_localize(None)

    # Keep min_date through the end of today. This also drops unreadable dates,
    # and mistyped future ones: those would otherwise be taken for the most
    # recent day below and skew every year-to-date figure.
    if today is None:
        today = pd.Timestamp.now(tz=local_timezone).date()
    tomorrow = pd.Timestamp(today) + pd.Timedelta(days=1)
    df = df[(df['incident_occurred_at'] >= min_date) & (df['incident_occurred_at'] < tomorrow)].copy()

    df['date'] = df['incident_occurred_at'].dt.date
    df['incident_year'] = df['incident_occurred_at'].dt.year

    # Trim padding so the same value is not counted under two spellings
    for column in TEXT_COLUMNS:
        if column in df.columns and pd.api.types.is_string_dtype(df[column].dtype):
            df[column] = df[column].str.strip()

    # Find the most recent date and exclude it (incomplete data)
    most_recent_date = df['date'].max()
    df = df[df['date'] < most_recent_date]

    # Filter out records with invalid coordinates
    df = df[(df['latitude'] != 0) & (df['longitude'] != 0)]
    df = df[(df['latitude'].between(lat_bounds[0], lat_bounds[1])) &
            (df['longitude'].between(lon_bounds[0], lon_bounds[1]))]

    return df


# ── Periods and daily counts ─────────────────────────────────────────────────

def selection_period(selected_year: Any, data_start: date, data_end: date) -> Tuple[date, date]:
    """
    First and last day covered by the Year filter.

    "All" is everything the dataset covers; a year is that calendar year, cut
    to the days the dataset covers. Rates and daily charts are measured over
    this period rather than from the first to the last incident of a
    selection, so the quiet days before and after still count.

    Args:
        selected_year: "All" or a year
        data_start: First day the dataset covers
        data_end: Last day the dataset covers
    """
    if selected_year == "All":
        return data_start, data_end
    year = int(selected_year)
    return max(date(year, 1, 1), data_start), min(date(year, 12, 31), data_end)


def daily_incident_counts(df: pd.DataFrame, start: date, end: date) -> pd.DataFrame:
    """
    Incidents per day for every day from `start` to `end`.

    Returns:
        DataFrame with a `date` (datetime) and a `count` column, one row per
        day, with 0 on the days that had no incidents
    """
    days = pd.date_range(start=start, end=end, freq="D", name="date")
    counts = df.groupby("date").size()
    counts.index = pd.to_datetime(counts.index)
    return counts.reindex(days, fill_value=0).reset_index(name="count")


# ── Year over year ───────────────────────────────────────────────────────────

def days_in_year(year: int) -> int:
    """Number of days in the given calendar year."""
    return 366 if calendar.isleap(year) else 365


def project_full_year(count_to_date: int, as_of: date) -> Optional[float]:
    """
    Scale a year-to-date count up to a full-year estimate.

    Args:
        count_to_date: Incidents from 1 January through `as_of`
        as_of: Last day the data covers. Using this rather than today's date
            keeps the estimate from being dragged down by the days between the
            end of the data and now (the feed lags by several days).

    Returns:
        Projected full-year count, or None until MIN_PROJECTION_DAYS of the
        year have data
    """
    days_elapsed = as_of.timetuple().tm_yday
    if days_elapsed < MIN_PROJECTION_DAYS:
        return None
    return count_to_date * days_in_year(as_of.year) / days_elapsed


def compute_yoy_deltas(df_filtered: pd.DataFrame, as_of: date) -> Tuple[Optional[str], Optional[str]]:
    """
    Compute year-over-year deltas for Total Incidents and Avg Daily.

    The current year is the year of `as_of` (the last day the data covers).
    Both years are measured over calendar days: days elapsed through `as_of`
    for the current year, and the full year for the prior one.

    Args:
        df_filtered: Incidents for the current filter selection
        as_of: Last day the full dataset covers

    Returns:
        (incidents_delta, avg_delta) as formatted strings, or None when a
        year has no incidents or too little of the current year has elapsed
    """
    current_year = as_of.year
    prior_year = current_year - 1

    counts_by_year = df_filtered["incident_year"].value_counts()
    cy_count = int(counts_by_year.get(current_year, 0))
    py_count = int(counts_by_year.get(prior_year, 0))
    if cy_count == 0 or py_count == 0:
        return None, None

    cy_projected = project_full_year(cy_count, as_of)
    if cy_projected is None:
        return None, None

    incidents_delta = f"{(cy_projected - py_count) / py_count * 100:+.1f}%"

    cy_avg = cy_count / as_of.timetuple().tm_yday
    py_avg = py_count / days_in_year(prior_year)
    avg_delta = f"{(cy_avg - py_avg) / py_avg * 100:+.1f}%"

    return incidents_delta, avg_delta
