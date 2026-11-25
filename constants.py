"""Constants for Detroit Crime Dashboard

This module contains all configuration constants, magic numbers, and hardcoded values
used throughout the Detroit crime incidents dashboard application.
"""

# Data source
DATA_URL = "https://data.detroitmi.gov/api/download/v1/items/8e532daeec1149879bd5e67fdd9c8be0/csv?layers=0"

# The feed publishes timestamps in UTC; all dates in the app are Detroit local time.
LOCAL_TIMEZONE = "America/Detroit"

# Data filtering
MIN_DATE = '2018-01-01'
DETROIT_LAT_MIN = 42.0
DETROIT_LAT_MAX = 43.0
DETROIT_LON_MIN = -84.0
DETROIT_LON_MAX = -82.0

# Forecasting parameters
# Days with at least one incident that a selection needs before it is forecast
MIN_FORECAST_DAYS = 30
FORECAST_PERIODS = 365  # 1 year
# Days of the current year needed before it is projected to a full year or
# compared with the year before
MIN_PROJECTION_DAYS = 60  # 2 months
HISTORICAL_MONTHS = 3  # 3 months of history in forecast chart

# Display parameters
CHART_HEIGHT = 400
TOP_N_ITEMS = 10
# Scatter mode plots at most this many points; wider selections are sampled.
# Every point is sent to the browser on each rerun, at about 110 bytes apiece,
# so this many is some 5 MB.
MAX_SCATTER_POINTS = 50_000
# Heatmap mode sends one record for each distinct position, at about 40 bytes
# apiece, so this many is some 4 MB. With more than this, positions are rounded
# more coarsely until they fit; every incident still counts towards its spot.
MAX_HEATMAP_CELLS = 100_000

# Share of timestamps that may be unreadable before the load is refused. A few
# bad rows are dropped; more than this means the feed itself has changed.
MAX_UNPARSEABLE_DATE_SHARE = 0.01

# The feed's last day is the latest one with at least this share of a typical
# (median) day's incidents. That day and anything after it is left out as
# incomplete; a lone mistyped row dated later does not count as a day.
MIN_LAST_DAY_SHARE = 0.1

# A refresh is refused, and the copy in use kept, if it has fewer than this
# share of that copy's incidents: the feed grows from day to day, so one that
# has shrunk this much is a short export or has lost its coordinates.
MIN_REFRESH_ROW_SHARE = 0.9

# Cache settings (in seconds)
DATA_CACHE_TTL = 86400  # 24 hours
# After a failed refresh the previous data stays in use; wait this long before
# trying the download again.
DATA_RETRY_INTERVAL = 900  # 15 minutes
# A failed first load leaves nothing to show, so it is tried again sooner.
# Until then every visitor is shown the failure rather than repeating the download.
DATA_FIRST_LOAD_RETRY_INTERVAL = 60  # 1 minute
# Both are for a failure that can pass (the network, a busy portal). One that
# would only repeat, such as a feed that has lost a column or a refresh refused
# as shrunken, waits twice as long after each such failure in a row, up to
# this: every try is a whole download. It is not longer because such a failure
# can pass after all: the portal empties the dataset for hours while it reloads
# it, and an export made then looks like a feed that has shrunk.
DATA_MAX_RETRY_INTERVAL = 21600  # 6 hours
# How often an open page that is waiting for a download asks whether it has
# ended, so that it can show the result without the user clicking anything.
DATA_POLL_INTERVAL = 5
# How often a page with nothing to wait for asks whether that is still so. A
# page left open follows a refresh that another visitor set off, and sets one
# off itself once the data is due, within this long.
DATA_IDLE_POLL_INTERVAL = 60
# How long the portal may say nothing before a download attempt is given up.
# The whole export arrives in seconds, so one that is silent this long has
# stalled.
DOWNLOAD_READ_TIMEOUT = 60
# No further attempt at a download is started once this long has passed since
# the first began. A first load is waited for, and until it ends its visitors
# see only a spinner: this bounds that to a few minutes with a stalled portal.
# A download that is still coming in is not cut off by it.
DOWNLOAD_RETRY_BUDGET = 120
FORECAST_CACHE_TTL = 3600  # 1 hour
YOY_CACHE_TTL = 1800  # 30 minutes

# Columns the app reads from the feed. Only these are loaded, and all of them
# must be present for the data to pass validation.
REQUIRED_COLUMNS = [
    'latitude',
    'longitude',
    'incident_occurred_at',
    'offense_category',
    'offense_description',
    'case_status',
    'police_precinct',
    'council_district',
    'neighborhood',
    'zip_code',
    'nearest_intersection',
    'incident_day_of_week',
    'incident_hour_of_day'
]

# Free-text columns that are whitespace-trimmed on load (the feed pads some
# values with trailing spaces, which would otherwise count as separate values).
TEXT_COLUMNS = [
    'offense_category',
    'offense_description',
    'case_status',
    'neighborhood',
    'nearest_intersection',
]

# Columns holding plain numbers (the local day of the week and hour of the
# day). They are read as numbers on load; a cell that is not one is missing.
NUMBER_COLUMNS = [
    'incident_day_of_week',
    'incident_hour_of_day',
]

# Columns holding numeric codes, with the prefix used to label each code
# (e.g. 3 -> "Precinct 3"; zip codes are shown as they are). They are read as
# whole numbers above zero on load; a cell that is not one is missing.
CODE_LABEL_PREFIXES = {
    'zip_code': '',
    'police_precinct': 'Precinct',
    'council_district': 'District',
}

# Holidays are named beside their marks on the forecast, where there is room
# for about this many characters. The official names that are longer have a
# short one here; any other long name is cut at a word.
MAX_HOLIDAY_LABEL_LENGTH = 16
HOLIDAY_SHORT_NAMES = {
    "Martin Luther King Jr. Day": "MLK Day",
    "Washington's Birthday": "Presidents' Day",
    "Juneteenth National Independence Day": "Juneteenth",
}

# Day of week labels. The feed numbers days ISO-style: 1=Monday ... 7=Sunday.
DAY_LABELS = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

# Available columns for pivot analysis
PIVOT_COLUMNS = [
    'offense_category',
    'case_status',
    'police_precinct',
    'council_district',
    'neighborhood',
    'zip_code',
    'incident_year',
    'incident_day_of_week',
    'incident_hour_of_day'
]

# Color schemes
COLOR_SCALE_HEATMAP = "RdYlGn_r"  # Red-Yellow-Green reversed (green=low, red=high)
