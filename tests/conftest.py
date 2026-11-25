"""Shared fixtures: a small synthetic copy of the Detroit crime feed."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from constants import (  # noqa: E402
    DETROIT_LAT_MAX, DETROIT_LAT_MIN, DETROIT_LON_MAX, DETROIT_LON_MIN, MIN_DATE,
)
from utils import clean_and_filter_data  # noqa: E402

# More categories than the map palette has colors (15), to exercise "Other"
CATEGORIES = [f"CATEGORY {i:02d}" for i in range(1, 21)]
STATUSES = ["ACTIVE", "INACTIVE", "WARRANT OBTAINED", "EXCEPTIONALLY CLEARED"]


def make_raw_feed(start="2024-06-01", end="2026-03-31 23:00", per_day=8, seed=0) -> pd.DataFrame:
    """Build rows shaped like the portal's CSV: UTC timestamps, local hour/day fields."""
    rng = np.random.default_rng(seed)
    days = pd.date_range(start, end, freq="D")
    n = len(days) * per_day

    # Local (Detroit) wall-clock times, then the UTC strings the feed publishes
    local = (
        pd.Series(np.repeat(days.values, per_day))
        + pd.to_timedelta(rng.integers(0, 24 * 60, n), unit="min")
    )
    utc = local.dt.tz_localize("America/Detroit", ambiguous=True, nonexistent="shift_forward").dt.tz_convert("UTC")

    # Popular categories first, so the ranking is stable
    weights = np.linspace(20, 1, len(CATEGORIES))
    category = rng.choice(CATEGORIES, n, p=weights / weights.sum())
    # Like the real feed: some descriptions are padded with trailing spaces
    padding = np.where(rng.random(n) < 0.5, " " * 20, "")

    return pd.DataFrame({
        "latitude": rng.uniform(42.30, 42.44, n),
        "longitude": rng.uniform(-83.25, -82.95, n),
        "incident_occurred_at": utc.dt.strftime("%Y/%m/%d %H:%M:%S+00"),
        "incident_entry_id": [f"{i}-1301" for i in range(n)],
        "offense_category": category,
        "offense_description": pd.Series(category) + " - DETAIL" + padding,
        "case_status": rng.choice(STATUSES, n),
        "police_precinct": rng.integers(2, 13, n),
        "council_district": rng.integers(1, 8, n),
        "neighborhood": rng.choice(["Greektown", "Nolan", "Rivertown", "Claytown"], n),
        "zip_code": rng.choice([48201, 48226, 48207, 48210], n),
        "nearest_intersection": rng.choice(["Brush St & Monroe St", "Cliff St & E 7 Mile Rd"], n),
        "incident_day_of_week": local.dt.day_of_week + 1,  # ISO: 1=Monday
        "incident_hour_of_day": local.dt.hour,
    })


@pytest.fixture(scope="session")
def raw_feed() -> pd.DataFrame:
    return make_raw_feed()


@pytest.fixture(scope="session")
def clean_df(raw_feed) -> pd.DataFrame:
    return clean_and_filter_data(
        raw_feed.copy(), MIN_DATE,
        (DETROIT_LAT_MIN, DETROIT_LAT_MAX), (DETROIT_LON_MIN, DETROIT_LON_MAX),
    )


@pytest.fixture
def page_state(clean_df) -> dict:
    """The session state app.py prepares before running a page."""
    return {
        "df_filtered": clean_df,
        "selected_category": "All",
        "category_order": clean_df["offense_category"].value_counts().index.tolist(),
        "data_end_date": clean_df["incident_occurred_at"].max().date(),
    }
