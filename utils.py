"""Utility functions for Detroit Crime Dashboard

This module provides common utility functions for data loading, validation,
and processing used throughout the crime dashboard application.
"""

import calendar
import logging
import tempfile
import threading
import time
import traceback
from datetime import date
from typing import IO, Any, Callable, Iterable, Optional, Tuple

import numpy as np
import pandas as pd
import requests
import streamlit as st
from urllib3.exceptions import ReadTimeoutError

from constants import (
    CODE_LABEL_PREFIXES, DATA_CACHE_TTL, DATA_FIRST_LOAD_RETRY_INTERVAL, DATA_MAX_RETRY_INTERVAL,
    DATA_RETRY_INTERVAL, DAY_LABELS, DOWNLOAD_ATTEMPT_TIMEOUT, DOWNLOAD_READ_TIMEOUT,
    DOWNLOAD_RETRY_BUDGET, HOLIDAY_SHORT_NAMES, LOCAL_TIMEZONE, MAX_FEED_LAG_DAYS,
    MAX_HOLIDAY_LABEL_LENGTH, MAX_REFRESH_DAYS_LOST, MAX_UNPARSEABLE_DATE_SHARE,
    MIN_LAST_DAY_SHARE, MIN_PROJECTION_DAYS, MIN_REFRESH_ROW_SHARE, NUMBER_COLUMNS,
    REQUIRED_COLUMNS, TEXT_COLUMNS,
)

logger = logging.getLogger(__name__)


def validate_data(df: Optional[pd.DataFrame]) -> bool:
    """Validate that the dataframe has required columns and basic data quality"""
    if df is None:
        raise ValueError("DataFrame is empty or None")

    # Check for required columns. This comes before the check for an empty
    # frame: one loaded with none of its columns is empty as well, and the
    # columns it lacks are what there is to say about it.
    missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    if df.empty:
        raise ValueError("DataFrame is empty or None")

    # Check for basic data quality
    if df['latitude'].isna().all() or df['longitude'].isna().all():
        raise ValueError("All latitude/longitude values are missing")

    if df['incident_occurred_at'].isna().all():
        raise ValueError("All incident dates are missing")

    return True


def whole_numbers(series: pd.Series) -> pd.Series:
    """Read a column as whole numbers, with a missing value for anything else.

    Anything else is text, fractions (3.7 is not hour 3), infinities, and
    cells that were empty to begin with. The result is of the integer type
    that allows for missing values, so one empty cell does not turn every
    other number into a float (3.0).
    """
    numbers = pd.to_numeric(series, errors='coerce')
    # Beyond 2**53 a float no longer holds a whole number exactly
    whole = ((numbers % 1 == 0) & (numbers.abs() < 2**53)).fillna(False).astype(bool)
    return numbers.where(whole).astype("Int64")


# Text written as a plain number: digits, with nothing but zeros after the point
_PLAIN_NUMBER = r"\s*\d+(?:\.0*)?\s*"


def numeric_codes(series: pd.Series) -> pd.Series:
    """Parse a column of numeric codes to ints, with 0 for anything invalid.

    Invalid is anything that is not a whole number above zero: text, missing
    values, fractions (3.7 is not precinct 3), infinities and negative
    numbers. So is a value that a parser takes for a number but that is not
    written as one, such as "1e3" or True.
    """
    if pd.api.types.is_bool_dtype(series.dtype):
        return pd.Series(0, index=series.index, dtype=int)
    codes = whole_numbers(series)
    valid = (codes > 0).fillna(False).astype(bool)
    if not pd.api.types.is_numeric_dtype(series.dtype):
        # Text, or text and numbers mixed
        valid &= series.astype(str).str.fullmatch(_PLAIN_NUMBER, na=False).astype(bool)
    return codes.where(valid, 0).astype(int)


def code_labels(codes: Iterable[int], prefix: str = "") -> list[str]:
    """Labels for numeric codes: 3 becomes "Precinct 3", or "3" with no prefix."""
    return [f"{prefix} {code}" if prefix else str(code) for code in codes]


def day_labels(days: Iterable[Any]) -> list[str]:
    """Names for the feed's ISO day numbers: 1 becomes "Monday", 7 "Sunday".

    Anything else is shown as it is, so an unexpected value stays visible. So
    is a missing value, which cannot be compared with the numbers.
    """
    return [
        DAY_LABELS[int(day) - 1] if not pd.isna(day) and day in range(1, 8) else str(day)
        for day in days
    ]


def holiday_label(name: str) -> str:
    """A holiday's name, short enough to sit beside its mark on the forecast.

    The long official names have a short one of their own. Any other name
    that is too long is cut at a word, so what is left can still be read.
    """
    short = HOLIDAY_SHORT_NAMES.get(name, name)
    if len(short) <= MAX_HOLIDAY_LABEL_LENGTH:
        return short
    kept = ""
    for word in short.split():
        longer = f"{kept} {word}".strip()
        if len(longer) >= MAX_HOLIDAY_LABEL_LENGTH:
            break
        kept = longer
    return (kept or short[:MAX_HOLIDAY_LABEL_LENGTH - 1]) + "…"


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
    """The portal answered, but not with the data (e.g. while it rebuilds the export).

    That is a status document in place of the CSV, or a CSV that is empty or
    has only its header: the portal empties the dataset while it reloads it.
    """


def is_transient_error(error: Exception) -> bool:
    """Whether a failed download is worth another attempt.

    Network trouble, rate limiting, server errors and a portal that is not
    ready are; a client error (such as 404) or a file that cannot be parsed
    will fail the same way every time.
    """
    if isinstance(error, requests.exceptions.HTTPError):
        return getattr(error.response, "status_code", None) in RETRY_STATUS_CODES
    return isinstance(error, TRANSIENT_ERRORS + (DataNotReadyError,))


def is_timeout(error: BaseException) -> bool:
    """Whether a failed download is the portal taking too long to say anything.

    `requests` reports a portal that is silent before its first byte as a
    Timeout. One that goes silent in the middle of the download it reports as
    a ConnectionError around urllib3's ReadTimeoutError, which would otherwise
    be taken for a portal that could not be reached.
    """
    if isinstance(error, requests.exceptions.Timeout):
        return True
    return (
        isinstance(error, requests.exceptions.ConnectionError)
        and bool(error.args)
        and isinstance(error.args[0], ReadTimeoutError)
    )


def _require_csv(start: bytes, content_type: str) -> None:
    """Raise DataNotReadyError when a response body is JSON or HTML, not CSV."""
    first = start.lstrip(b"\xef\xbb\xbf \t\r\n")[:1]
    if first in (b"{", b"[", b"<"):
        raise DataNotReadyError(
            f"the portal sent {content_type or 'an unlabelled response'} instead of the CSV"
        )


def _has_rows(csv_file: IO[bytes]) -> bool:
    """Whether a CSV file holds anything after its first line, the header."""
    csv_file.seek(0)
    csv_file.readline()
    return any(line.strip() for line in csv_file)


# The download is read in pieces of at most this many bytes. The time an
# attempt has taken is looked at as each piece arrives, so they are kept small
# enough to arrive often on a slow line.
DOWNLOAD_CHUNK_SIZE = 64 * 1024


def safe_load_data_from_url(url: str, timeout: int = DOWNLOAD_READ_TIMEOUT, max_retries: int = 5,
                            usecols: Optional[Iterable[str]] = None,
                            retry_budget: float = DOWNLOAD_RETRY_BUDGET,
                            attempt_timeout: float = DOWNLOAD_ATTEMPT_TIMEOUT) -> pd.DataFrame:
    """
    Safely load data from URL with application-level retries.

    Retries the entire download on transient failures (see is_transient_error),
    including IncompleteRead errors that occur mid-stream during large
    downloads. Failures that would only repeat, such as a 404 or a file that
    cannot be parsed, are raised straight away.

    The download is spooled to a temporary file and parsed from there, so the
    only large object held in memory is the resulting DataFrame.

    This runs on the dataset cache's own thread, not a session's, so it makes
    no Streamlit calls: retries go to the log, and the page that is waiting
    shows its own spinner.

    Args:
        url: URL to fetch data from
        timeout: Read timeout in seconds: how long the portal may say nothing
            before the attempt is given up
        max_retries: Number of full download attempts (at least 1)
        usecols: Columns to load; others are skipped. Columns missing from the
            feed are ignored here so validation can report them by name.
        retry_budget: Seconds from the start of the first attempt after which
            no further attempt is started. With `timeout` it bounds how long a
            stalled portal is waited for; a download that is still coming in
            is not cut off by it.
        attempt_timeout: Seconds one attempt may take from start to finish.
            A download that is still coming in after that is given up as too
            slow (a Timeout): `timeout` only counts silence, and would let
            one that trickles in run for ever, with every later load waiting
            behind it. It is looked at as each piece of the download arrives,
            so an attempt overruns it by the time its last piece took.

    Returns:
        DataFrame

    Raises:
        DataNotReadyError: the portal answered without the CSV, or with one
            that is empty or has no rows
        requests.exceptions.RequestException: the download failed, or took
            longer than `attempt_timeout`
        ValueError: the file could not be parsed
    """
    if max_retries < 1:
        raise ValueError("max_retries must be at least 1")

    wanted = set(usecols) if usecols is not None else None
    column_filter = (lambda name: name in wanted) if wanted is not None else None

    started = time.monotonic()
    attempt = 0
    while True:
        attempt += 1
        attempt_started = time.monotonic()
        try:
            with tempfile.TemporaryFile() as buffer:
                with requests.get(url, timeout=(15, timeout), stream=True) as response:
                    response.raise_for_status()
                    # e.g. 202 while the portal is still building the export
                    if response.status_code != 200:
                        raise DataNotReadyError(
                            f"the portal answered with status {response.status_code} instead of the data"
                        )
                    checked = False
                    for chunk in response.iter_content(chunk_size=DOWNLOAD_CHUNK_SIZE):
                        # Still coming in, but too slowly ever to be worth the wait
                        if time.monotonic() - attempt_started > attempt_timeout:
                            raise requests.exceptions.Timeout(
                                f"the download was still coming in after {attempt_timeout:,.0f} seconds"
                            )
                        if not chunk:
                            continue
                        if not checked:
                            _require_csv(chunk, response.headers.get("Content-Type", ""))
                            checked = True
                        buffer.write(chunk)

                buffer.seek(0)
                try:
                    # A stray byte that is not UTF-8 costs one character, not the dataset
                    df = pd.read_csv(
                        buffer, encoding="utf-8-sig", encoding_errors="replace",
                        usecols=column_filter, low_memory=False,
                    )
                except pd.errors.EmptyDataError:
                    raise DataNotReadyError("the portal sent an empty file instead of the CSV") from None
                # A file with no rows is an export caught while the portal reloads
                # its data, not a feed that has changed: worth another attempt. No
                # rows here can also mean none of the wanted columns (which is
                # for validation to report), so the file itself is asked.
                if len(df) == 0 and not _has_rows(buffer):
                    raise DataNotReadyError("the portal sent a file with no rows in it")
                return df

        except Exception as e:
            # Attempts that each waited out a silent portal add up: one that
            # could only start once the budget is spent (the wait before it
            # counts) is not made, and the failure is reported instead
            wait = 2 ** attempt
            out_of_time = time.monotonic() - started + wait >= retry_budget
            if attempt >= max_retries or out_of_time or not is_transient_error(e):
                raise
            logger.warning(
                "Download attempt %d of %d failed, trying again in %ds: %s", attempt, max_retries, wait, e
            )
            time.sleep(wait)


def _release_frames(error: BaseException) -> None:
    """Drop the tracebacks held by an exception and by the ones it was raised from.

    A kept exception otherwise keeps every frame of the failed call alive, and
    with them whatever they were holding: here, a half-loaded dataset.
    """
    pending, seen = [error], set()
    while pending:
        current = pending.pop()
        if current is None or id(current) in seen:
            continue
        seen.add(id(current))
        traceback.clear_frames(current.__traceback__)
        current.__traceback__ = None
        pending += [current.__cause__, current.__context__]


class _Load:
    """One run of a loader: how it ended, and an event for those waiting on it."""

    def __init__(self) -> None:
        self.done = threading.Event()
        self.value: Any = None
        self.error: Optional[BaseException] = None
        self.version = 0  # the cache's version once this load had ended


class LoadFailedError(Exception):
    """A load failed and there is no earlier value to use in its place.

    Raised by `RefreshingCache.get`, a new one for each caller. The failure
    itself, which every caller is told of, is its `__cause__`: that one object
    is shared between threads, so it is never raised again. `version` is the
    cache's version at that failure (see `RefreshingCache.get_versioned`).
    """

    def __init__(self, message: str, version: int = 0) -> None:
        super().__init__(message)
        self.version = version


class RefreshingCache:
    """Keeps the last good result of an expensive load and refreshes it when stale.

    Unlike a cache with a time-to-live, the old value is not thrown away when
    it expires: it stays in use until a refresh succeeds. If a refresh fails,
    callers get the old value together with the error, and the load is not
    tried again until `retry_interval` has passed.

    A failure that would only repeat (see `can_pass`) is left alone for longer
    each time: the wait doubles with every such failure in a row, up to
    `max_retry_interval`. Each try can cost a whole download, and waiting a
    quarter of an hour does not bring back a column the feed has lost.

    Loads run one at a time on a thread of the cache's own. So a load always
    runs to the end and is kept, even when the caller that started it is
    interrupted (Streamlit stops a session's script as soon as its user clicks
    something), and callers that arrive during a load share it rather than
    start their own. Only the first load is waited for: a refresh runs behind
    the old value, which every caller, the one that set it off included, gets
    straight away. The new value is returned once the refresh has ended.

    The cache counts the loads that have ended: its version. What `get`
    returns can only change when a load ends, so a caller that keeps the
    version it was given (see `get_versioned`) can ask later, with `at_rest`,
    whether calling `get` again would come to anything new, without keeping
    hold of the value itself.

    Args:
        ttl: Age at which the value is due for a refresh
        retry_interval: How long a failed refresh is left alone
        first_retry_interval: The same for a failed first load, which leaves
            nothing to show and so is usually tried again sooner. Defaults to
            `retry_interval`.
        clock: Source of the current time, in seconds. Defaults to the wall
            clock, which goes on counting while the machine is asleep: by a
            monotonic clock, which on Linux stands still then, a value loaded
            before a night's sleep is hours younger than it is. The wall
            clock can be set back; a value or a failure that it dates in the
            future is taken to be due.
        check: Called on the cache's thread with (new value, value in use)
            when a refresh has loaded. If it raises, the new value is refused
            and the refresh counts as failed. A first load has nothing to be
            compared with, so it is not checked.
        can_pass: Says whether a failure is one that can pass by itself, such
            as a fault on the network. One that cannot would only repeat, and
            is tried again less and less often (see above). Without it every
            failure is taken for one that can pass.
        max_retry_interval: The longest a failure that would only repeat is
            left alone. Defaults to `ttl`. What looks set to repeat can still
            pass (a short export while the portal reloads its data looks like
            a feed that has shrunk), and this is how long the cache can then
            take to find out.
    """

    def __init__(self, ttl: float, retry_interval: float,
                 first_retry_interval: Optional[float] = None,
                 clock: Callable[[], float] = time.time,
                 check: Optional[Callable[[Any, Any], None]] = None,
                 can_pass: Optional[Callable[[BaseException], bool]] = None,
                 max_retry_interval: Optional[float] = None) -> None:
        self._ttl = ttl
        self._retry_interval = retry_interval
        self._first_retry_interval = retry_interval if first_retry_interval is None else first_retry_interval
        self._max_retry_interval = ttl if max_retry_interval is None else max_retry_interval
        self._clock = clock
        self._check = check
        self._can_pass = can_pass
        # Guards the fields below. It is never held while a load runs.
        self._lock = threading.Lock()
        self._has_value = False
        self._value: Any = None
        self._loaded_at = 0.0
        self._failed_at: Optional[float] = None
        self._error: Optional[BaseException] = None
        self._retry_wait = 0.0  # how long the last failure is left alone
        self._repeating = False  # whether that failure was one that would only repeat
        self._load: Optional[_Load] = None  # the load under way, if any
        self._version = 0  # how many loads have ended

    def get(self, loader: Callable[[], Any]) -> Tuple[Any, Optional[BaseException]]:
        """Return (value, error), loading or refreshing with `loader` when due.

        `error` is None when the value is current, or is being refreshed for
        the first time since it was. When a refresh has failed it is that
        failure, and the value is the previous one. A refresh is not waited
        for: the previous value is returned while it runs.

        With no previous value there is nothing to return, so the load is
        waited for, and if it fails LoadFailedError is raised: for every
        caller that waited, and for those that come after until it is due
        another try.

        Raises:
            LoadFailedError: the load failed with no earlier value to use;
                its `__cause__` is the failure
        """
        value, error, _ = self.get_versioned(loader)
        return value, error

    def get_versioned(self, loader: Callable[[], Any]) -> Tuple[Any, Optional[BaseException], int]:
        """As `get`, with the cache's version as a third item: (value, error, version).

        The version is the one the value and error stand at: the number of
        loads that had ended when they were handed out. Keep it for `at_rest`.

        Raises:
            LoadFailedError: as for `get`; the version is its `version`
        """
        with self._lock:
            if self._fresh():
                return self._value, None, self._version

            load = self._load
            if load is None:
                if self._left_alone():
                    if self._has_value:
                        return self._value, self._error, self._version
                    raise LoadFailedError(str(self._error), self._version) from self._error
                load = _Load()
                threading.Thread(
                    target=self._run, args=(loader, load), name="refreshing-cache-load", daemon=True
                ).start()
                self._load = load

            # A refresh is under way: use what there is rather than wait
            if self._has_value:
                return self._value, self._error, self._version

        load.done.wait()
        if load.error is None:
            return load.value, None, load.version
        with self._lock:
            if self._has_value:
                return self._value, load.error, load.version
        raise LoadFailedError(str(load.error), load.version) from load.error

    def _fresh(self) -> bool:
        """Whether there is a value that is not yet due for a refresh. Called with the lock held.

        One loaded later than now (the clock has been set back) is due: how
        old it really is cannot be told any more.
        """
        return self._has_value and 0 <= self._clock() - self._loaded_at < self._ttl

    def _left_alone(self) -> bool:
        """Whether the last load failed too recently to be tried again. Called with the lock held.

        As for a value, a failure later than now is not left alone.
        """
        return self._failed_at is not None and 0 <= self._clock() - self._failed_at < self._retry_wait

    def _run(self, loader: Callable[[], Any], load: _Load) -> None:
        """Run one load on the cache's thread and record how it ended."""
        try:
            try:
                # Handed straight on, not kept in a variable here: this frame
                # is still running when a failure is logged, so a record of
                # that failure would keep a refused value alive through it
                load.value = self._checked(loader())
            except BaseException as error:  # whatever stops the load is its outcome
                load.error = error
                # The pages only show the message. This is the one record of
                # where it went wrong: the frames are let go of straight after.
                logger.exception("Loading failed")
                _release_frames(error)
        finally:
            # Whatever went wrong above, nobody is left waiting for this load
            # and the next one is free to start
            try:
                with self._lock:
                    self._load = None
                    self._version += 1
                    load.version = self._version
                    # Timed from the end of the attempt: a failure that was slow
                    # to show itself still earns the full wait before the next try
                    now = self._clock()
                    if load.error is None:
                        self._value, self._has_value = load.value, True
                        self._loaded_at, self._failed_at, self._error = now, None, None
                        self._repeating = False
                    else:
                        self._failed_at, self._error = now, load.error
                        self._retry_wait = self._wait_after(load.error)
            finally:
                load.done.set()

    def _wait_after(self, error: BaseException) -> float:
        """How long the loader is left alone after failing with `error`.

        The usual interval, or twice the last wait (up to `max_retry_interval`)
        when this failure and the one before it would both only repeat. Called
        with the lock held.
        """
        usual = self._retry_interval if self._has_value else self._first_retry_interval
        would_repeat = self._can_pass is not None and not self._can_pass(error)
        again = would_repeat and self._repeating
        self._repeating = would_repeat
        return min(self._retry_wait * 2, max(self._max_retry_interval, usual)) if again else usual

    def _checked(self, value: Any) -> Any:
        """Return a newly loaded value once `check` has passed it.

        It is put to `check` against the value in use. With none in use (a
        first load) there is nothing to compare it with, and it is returned
        as it is.
        """
        if self._check is not None:
            with self._lock:
                has_previous, previous = self._has_value, self._value
            if has_previous:
                self._check(value, previous)
        return value

    def wait(self, timeout: Optional[float] = None) -> bool:
        """Wait for the load under way, if there is one, to end.

        Returns:
            False if `timeout` (in seconds) passed first, True otherwise
        """
        with self._lock:
            load = self._load
        return load is None or load.done.wait(timeout)

    def at_rest(self, version: int) -> bool:
        """Whether `get` would do now just what it did at `version`, and start nothing.

        False once a load has ended since that version was handed out, while
        one is under way, and when the next `get` would start one: the value
        is due for a refresh, or a failure has been left alone for long
        enough. Asking only whether a load is running would miss a refresh
        that ended in between, and leave the caller with the old value and no
        reason to come back for the new one.
        """
        with self._lock:
            return (
                self._version == version
                and self._load is None
                and (self._fresh() or self._left_alone())
            )

    def expire(self) -> None:
        """Make the next `get` load again, whatever the age of the value or of a failure.

        As with any refresh, that `get` still returns the value there is.
        """
        with self._lock:
            self._loaded_at = self._clock() - self._ttl
            self._failed_at = None


def refuse_shrunken_refresh(new: pd.DataFrame, previous: pd.DataFrame) -> None:
    """Refuse a refreshed dataset that has lost incidents, or its latest days.

    From one day to the next the feed grows, and ends a day later. One that
    comes back with fewer incidents than the copy in use is a short export,
    or has lost the coordinates its rows are kept by; one that ends days
    earlier has been cut short by date, which can cost it only a few of its
    rows. Taking either would understate the figures without a word. Refused,
    the copy in use stays and the pages say that the refresh failed.

    What to do if the feed really has shrunk is written to the log, for
    whoever runs the app. The message raised is shown to every visitor, who
    can do nothing about it.

    Args:
        new: The cleaned dataset the refresh has loaded
        previous: The cleaned dataset in use

    Raises:
        ValueError: `new` has fewer than MIN_REFRESH_ROW_SHARE of the rows of
            `previous`, or ends more than MAX_REFRESH_DAYS_LOST days before it
    """
    if len(new) < len(previous) * MIN_REFRESH_ROW_SHARE:
        logger.warning(
            "Refusing a refresh with %d usable incidents, against %d in the copy in use. "
            "If the feed really has shrunk, restart the app to accept it.",
            len(new), len(previous),
        )
        raise ValueError(
            f"the refreshed feed has only {len(new):,} usable incidents, against {len(previous):,} "
            "in the copy shown; it may be incomplete or have lost its coordinates"
        )

    new_end, previous_end = new["date"].max(), previous["date"].max()
    days_lost = (previous_end - new_end).days
    if days_lost > MAX_REFRESH_DAYS_LOST:
        logger.warning(
            "Refusing a refresh that ends on %s, against %s in the copy in use. "
            "If the feed really has lost those days, restart the app to accept it.",
            new_end, previous_end,
        )
        raise ValueError(
            f"the refreshed feed ends on {new_end}, {days_lost} days before the copy shown; "
            "it may be incomplete"
        )


def local_today(timezone: str = LOCAL_TIMEZONE) -> date:
    """Today's date where the incidents happen, whatever the server's own clock is set to."""
    return pd.Timestamp.now(tz=timezone).date()


def refuse_stale_feed(df: pd.DataFrame, today: Optional[date] = None) -> None:
    """Refuse a cleaned dataset that ends long before today.

    The export runs a few days behind. One whose last incidents are weeks
    old has been cut short (a partial export, made while the portal reloads
    its data, can hold everything but the latest months), or the portal has
    stopped updating it. Shown as it is, the year so far would be measured up
    to a day long past and set against a last year that is complete.

    This is the one check a first load can be put to: there is no earlier
    copy to hold it against (see `refuse_shrunken_refresh`). It cannot tell
    an export that is thin throughout from a quieter city.

    What to do if the portal really has stopped is written to the log, for
    whoever runs the app. The message raised is shown to every visitor.

    Args:
        df: Cleaned dataset, with the local day of each incident in `date`
        today: Current local date. Defaults to `local_today()`.

    Raises:
        ValueError: the last day in `df` is more than MAX_FEED_LAG_DAYS
            before `today`
    """
    if today is None:
        today = local_today()
    last_day = df["date"].max()
    days_behind = (today - last_day).days
    if days_behind > MAX_FEED_LAG_DAYS:
        logger.warning(
            "Refusing a feed that ends on %s, %d days ago. If the portal has stopped "
            "updating it, raise MAX_FEED_LAG_DAYS in constants.py to show it as it is.",
            last_day, days_behind,
        )
        raise ValueError(
            f"the feed ends on {last_day}, {days_behind} days ago; it may be incomplete, "
            "or the portal may have stopped updating it"
        )


@st.cache_resource(show_spinner=False)
def dataset_cache() -> RefreshingCache:
    """The one dataset cache shared by every session in this process.

    The DataFrame it holds is shared too. Treat it as read-only: filter or
    copy it before changing anything.
    """
    return RefreshingCache(
        ttl=DATA_CACHE_TTL,
        retry_interval=DATA_RETRY_INTERVAL,
        first_retry_interval=DATA_FIRST_LOAD_RETRY_INTERVAL,
        check=refuse_shrunken_refresh,
        can_pass=is_transient_error,
        max_retry_interval=DATA_MAX_RETRY_INTERVAL,
    )


# ── Cleaning ─────────────────────────────────────────────────────────────────

def parse_feed_timestamps(raw: pd.Series) -> pd.Series:
    """
    Parse the feed's UTC timestamps, tolerating a few bad values.

    Values that cannot be read become NaT, and are dropped later along with
    the rows that have no date at all. If more than MAX_UNPARSEABLE_DATE_SHARE
    of them cannot be read the feed itself has changed, and dropping that many
    incidents would understate every count, so that raises instead.

    A cell of nothing but padding has no date: it is missing, not unreadable,
    as it is in the text columns.

    Raises:
        ValueError: too many timestamps could not be read
    """
    if pd.api.types.is_string_dtype(raw):
        raw = raw.mask(raw.str.strip() == "")

    try:
        occurred = pd.to_datetime(raw, utc=True, errors='coerce')
        unread = occurred.isna() & raw.notna()
        if unread.any():
            # Not written like the rest of the column: read those one by one.
            # They can come back in a finer unit than the column holds (a value
            # with nanoseconds does), which the column would refuse.
            reread = pd.to_datetime(raw[unread], utc=True, errors='coerce', format='mixed')
            occurred[unread] = reread.dt.as_unit(occurred.dt.unit)
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
    # One non-numeric cell makes read_csv hand over a whole coordinate column
    # as text. Read the coordinates as numbers, so that cell is dropped below
    # like a missing one instead of failing the comparisons further down.
    df = df.assign(
        latitude=pd.to_numeric(df['latitude'], errors='coerce'),
        longitude=pd.to_numeric(df['longitude'], errors='coerce'),
    ).dropna(subset=['latitude', 'longitude'])

    # The same goes for the feed's day and hour numbers. Left as text they
    # match none of the days and hours the charts lay out, and the heatmap
    # comes out empty; a cell that is not a whole number is missing instead.
    for column in NUMBER_COLUMNS:
        if column in df.columns:
            df[column] = whole_numbers(df[column])

    # Precinct, district and zip codes are read here as well, once, so the
    # pages only count and label them: reading one that arrived as text can
    # take a second at the feed's size, which a page would spend again on
    # every rerun. A cell that is not a valid code is missing.
    for column in CODE_LABEL_PREFIXES:
        if column in df.columns:
            codes = numeric_codes(df[column])
            df[column] = codes.where(codes != 0).astype("Int64")

    occurred = parse_feed_timestamps(df['incident_occurred_at'])
    df['incident_occurred_at'] = occurred.dt.tz_convert(local_timezone).dt.tz_localize(None)

    # Keep min_date through the end of today. This also drops unreadable dates,
    # and mistyped future ones: those would otherwise be taken for the most
    # recent day below and skew every year-to-date figure.
    if today is None:
        today = local_today(local_timezone)
    tomorrow = pd.Timestamp(today) + pd.Timedelta(days=1)
    df = df[(df['incident_occurred_at'] >= min_date) & (df['incident_occurred_at'] < tomorrow)].copy()

    df['date'] = df['incident_occurred_at'].dt.date
    df['incident_year'] = df['incident_occurred_at'].dt.year

    # Trim padding so the same value is not counted under two spellings. A
    # value that was only padding is missing, like an empty field in the CSV.
    for column in TEXT_COLUMNS:
        if column in df.columns and pd.api.types.is_string_dtype(df[column].dtype):
            stripped = df[column].str.strip()
            df[column] = stripped.mask(stripped == "")

    # The feed's last day is still being filled in, so it is left out, along
    # with anything dated after it. A stray row with a later date (a typo) must
    # not pass for that day, or the real one would be kept as if complete: the
    # last day is the latest one with a fair share of a typical day's incidents.
    # A real last day with fewer than that costs one complete day as well.
    per_day = df['date'].value_counts()
    if not per_day.empty:
        full_enough = per_day[per_day >= per_day.median() * MIN_LAST_DAY_SHARE]
        df = df[df['date'] < full_enough.index.max()]

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


def partial_years(data_start: date, data_end: date) -> dict[int, Tuple[date, date]]:
    """
    Years the dataset covers only in part, with the days of each it does cover.

    Those are its first year, unless it starts on 1 January, and its last,
    unless it ends on 31 December. Their counts are not those of a whole year,
    so they are marked wherever years are set side by side.

    Returns:
        {year: (first day covered, last day covered)}
    """
    partial = {}
    for year in {data_start.year, data_end.year}:
        covered = selection_period(year, data_start, data_end)
        if covered != (date(year, 1, 1), date(year, 12, 31)):
            partial[year] = covered
    return partial


def part_year_note(first: date, last: date) -> str:
    """Which part of its year a span of days is, in a few words: "to 9 Feb", "from 1 Jun"."""

    def day_month(day: date) -> str:
        return f"{day.day} {calendar.month_abbr[day.month]}"

    if (first.month, first.day) == (1, 1):
        return f"to {day_month(last)}"
    if (last.month, last.day) == (12, 31):
        return f"from {day_month(first)}"
    return f"{day_month(first)} to {day_month(last)}"


def project_full_year(count_to_date: int, as_of: date, data_start: date) -> Optional[float]:
    """
    Scale a year-to-date count up to a full-year estimate.

    Args:
        count_to_date: Incidents in the year of `as_of`, through that day
        as_of: Last day the data covers. Using this rather than today's date
            keeps the estimate from being dragged down by the days between the
            end of the data and now (the feed lags by several days).
        data_start: First day the data covers. When that is within the year,
            the count is only of the days since then, and it is scaled up from
            those rather than from the days since 1 January.

    Returns:
        Projected full-year count, or None until MIN_PROJECTION_DAYS of the
        year have data
    """
    first_day = max(date(as_of.year, 1, 1), data_start)
    days_covered = (as_of - first_day).days + 1
    if days_covered < MIN_PROJECTION_DAYS:
        return None
    return count_to_date * days_in_year(as_of.year) / days_covered


def same_day_last_year(day: date) -> date:
    """The same calendar day one year earlier; 29 February becomes 28 February."""
    try:
        return day.replace(year=day.year - 1)
    except ValueError:
        return day.replace(year=day.year - 1, day=28)


# A change that rounds to nothing. It carries no sign, so that it is not drawn
# as a rise or a fall (see `format_change`).
NO_CHANGE = "0.0%"


def format_daily_rate(per_day: float) -> str:
    """Incidents per day for the KPI card.

    One decimal place, or two significant digits below one a day, so a rare
    category shows as 0.0043 rather than 0.0. A rate that rounds up to one a
    day is shown as 1.0, like any other at or above one, and a very small one
    is written out in full rather than as 9e-05.
    """
    if per_day < 1:
        text = np.format_float_positional(per_day, precision=2, unique=False, fractional=False, trim="-")
        if float(text) < 1:
            return text
    return f"{per_day:.1f}"


def format_change(fraction: float) -> str:
    """A change as a signed percentage to one decimal place: "+4.2%", "-0.3%".

    One that rounds to zero is NO_CHANGE, without a sign: "+0.0%" would be
    taken for a rise and "-0.0%" for a fall by anything that goes by the sign.
    """
    text = f"{fraction * 100:+.1f}%"
    return NO_CHANGE if text.lstrip("+-") == NO_CHANGE else text


def compute_yoy_deltas(df_filtered: pd.DataFrame, as_of: date,
                       data_start: date) -> Tuple[Optional[str], Optional[str]]:
    """
    Compute year-over-year deltas for Total Incidents and Avg Daily.

    The current year is the year of `as_of` (the last day the data covers).
    It is compared like for like: 1 January through `as_of` against the same
    days of the year before. Measuring the year so far against the whole of
    the last one would show the seasons as a change (a quiet winter as a
    fall), because the months still to come are not the same as those past.

    The two deltas differ only around a leap day, when one of the two spans
    is a day longer: the total counts that day, the daily average does not.

    Args:
        df_filtered: Incidents for the current filter selection, with their
            local time in `incident_occurred_at`
        as_of: Last day the full dataset covers
        data_start: First day the full dataset covers

    Returns:
        (incidents_delta, avg_delta) as formatted strings (see
        `format_change`), or None when either span has no incidents, when
        fewer than MIN_PROJECTION_DAYS of the current year have data, or when
        the dataset does not reach back to 1 January of the year before: the
        days it lacks would be counted as days without incidents, and show as
        a rise.
    """
    cy_days = as_of.timetuple().tm_yday
    if cy_days < MIN_PROJECTION_DAYS:
        return None, None
    prior_as_of = same_day_last_year(as_of)
    if data_start > date(prior_as_of.year, 1, 1):
        return None, None
    py_days = prior_as_of.timetuple().tm_yday

    def incidents_through(last_day: date) -> int:
        """Incidents from 1 January of `last_day`'s year through that day."""
        occurred = df_filtered["incident_occurred_at"]
        start = pd.Timestamp(last_day.year, 1, 1)
        end = pd.Timestamp(last_day) + pd.Timedelta(days=1)
        return int(((occurred >= start) & (occurred < end)).sum())

    cy_count = incidents_through(as_of)
    py_count = incidents_through(prior_as_of)
    if cy_count == 0 or py_count == 0:
        return None, None

    incidents_delta = format_change((cy_count - py_count) / py_count)

    cy_avg = cy_count / cy_days
    py_avg = py_count / py_days
    avg_delta = format_change((cy_avg - py_avg) / py_avg)

    return incidents_delta, avg_delta
