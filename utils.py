"""Utility functions for Detroit Crime Dashboard

This module provides common utility functions for data loading, validation,
and processing used throughout the crime dashboard application.
"""

import calendar
import tempfile
import time
from datetime import date
from typing import Iterable, Optional, Tuple

import pandas as pd
import requests
import streamlit as st

from constants import LOCAL_TIMEZONE, MIN_PROJECTION_DAYS, REQUIRED_COLUMNS, TEXT_COLUMNS


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


def _numeric_codes(series: pd.Series) -> pd.Series:
    """Parse a column of numeric codes to ints, with 0 for anything invalid."""
    return pd.to_numeric(series, errors='coerce').fillna(0).astype(int)


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
    codes = _numeric_codes(series)
    labels = codes[codes != 0].astype(str)
    return f"{prefix} " + labels if prefix else labels


def format_categorical_column(df: pd.DataFrame, column_name: str, prefix: str = "") -> pd.DataFrame:
    """
    Standardize categorical column formatting to remove decimals

    Args:
        df: DataFrame to process
        column_name: Name of column to format
        prefix: Optional prefix to add (e.g., "Precinct", "District")

    Returns:
        Copy of the DataFrame with the column formatted and invalid rows removed
    """
    if column_name not in df.columns:
        return df.copy()

    df_copy = df[_numeric_codes(df[column_name]) != 0].copy()
    df_copy[column_name] = format_categorical_series(df_copy[column_name], prefix)

    return df_copy


def format_zip_code_column(df: pd.DataFrame, column_name: str = 'zip_code') -> pd.DataFrame:
    """
    Format zip code column using string replacement method

    Args:
        df: DataFrame to process
        column_name: Name of zip code column

    Returns:
        DataFrame with formatted zip codes
    """
    df_copy = df.copy()

    if column_name not in df_copy.columns:
        return df_copy

    # More robust zip code formatting using string replacement
    df_copy[column_name] = df_copy[column_name].astype(str).str.replace('.0', '').str.replace('.', '')

    # Remove invalid entries
    invalid_values = ['0', 'nan', '', 'None']
    df_copy = df_copy[~df_copy[column_name].isin(invalid_values)]

    return df_copy


def safe_load_data_from_url(url: str, timeout: int = 300, max_retries: int = 5,
                            usecols: Optional[Iterable[str]] = None) -> pd.DataFrame:
    """
    Safely load data from URL with application-level retries.

    Retries the entire download on any transient failure including
    IncompleteRead errors that occur mid-stream during large downloads.

    The download is spooled to a temporary file and parsed from there, so the
    only large object held in memory is the resulting DataFrame.

    Args:
        url: URL to fetch data from
        timeout: Read timeout in seconds
        max_retries: Number of full download attempts
        usecols: Columns to load; others are skipped. Columns missing from the
            feed are ignored here so validation can report them by name.

    Returns:
        DataFrame

    Raises:
        Various specific exceptions for different failure modes
    """
    wanted = set(usecols) if usecols is not None else None
    column_filter = (lambda name: name in wanted) if wanted is not None else None

    last_error = None
    status_placeholder = st.empty()
    for attempt in range(1, max_retries + 1):
        try:
            with tempfile.TemporaryFile() as buffer:
                with st.spinner(f"Downloading Detroit crime data (attempt {attempt}/{max_retries})..."):
                    with requests.get(url, timeout=(15, timeout), stream=True) as response:
                        response.raise_for_status()
                        for chunk in response.iter_content(chunk_size=1024 * 1024):
                            if chunk:
                                buffer.write(chunk)

                buffer.seek(0)
                df = pd.read_csv(buffer, encoding="utf-8-sig", usecols=column_filter, low_memory=False)

            status_placeholder.empty()
            return df

        except requests.exceptions.Timeout:
            status_placeholder.empty()
            raise
        except requests.exceptions.HTTPError:
            status_placeholder.empty()
            raise
        except pd.errors.ParserError:
            status_placeholder.empty()
            raise
        except Exception as e:
            last_error = e
            if attempt < max_retries:
                wait = 2 ** attempt
                status_placeholder.warning(f"Download interrupted, retrying in {wait}s... ({e})")
                time.sleep(wait)
                continue
            status_placeholder.empty()
            raise

    raise last_error


def clean_and_filter_data(df: pd.DataFrame, min_date: str,
                         lat_bounds: Tuple[float, float],
                         lon_bounds: Tuple[float, float],
                         local_timezone: str = LOCAL_TIMEZONE) -> pd.DataFrame:
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

    Returns:
        Cleaned DataFrame
    """
    # Clean and prepare data
    df = df.dropna(subset=['latitude', 'longitude'])

    # Convert date with error handling
    try:
        occurred = pd.to_datetime(df['incident_occurred_at'], utc=True)
    except (ValueError, TypeError) as e:
        raise ValueError(f"Failed to parse incident dates: {str(e)}")

    df['incident_occurred_at'] = occurred.dt.tz_convert(local_timezone).dt.tz_localize(None)

    # Filter for data from min_date forward (this also drops unparseable dates)
    df = df[df['incident_occurred_at'] >= min_date].copy()

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
