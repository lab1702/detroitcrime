"""Constants for Detroit Crime Dashboard

This module contains all configuration constants, magic numbers, and hardcoded values
used throughout the Detroit crime incidents dashboard application.
"""

# Data source
DATA_URL = "https://data.detroitmi.gov/api/download/v1/items/8e532daeec1149879bd5e67fdd9c8be0/csv?layers=0"

# Data filtering
MIN_DATE = '2018-01-01'
DETROIT_LAT_MIN = 42.0
DETROIT_LAT_MAX = 43.0
DETROIT_LON_MIN = -84.0
DETROIT_LON_MAX = -82.0

# Forecasting parameters
MIN_FORECAST_DAYS = 30
FORECAST_PERIODS = 365  # 1 year
MIN_PROJECTION_DAYS = 60  # 2 months
HISTORICAL_MONTHS = 3  # 3 months of history in forecast chart

# Display parameters
CHART_HEIGHT = 400
TOP_N_ITEMS = 10
MAX_MAP_SAMPLE = 1000

# Cache settings (in seconds)
DATA_CACHE_TTL = 86400  # 24 hours
FORECAST_CACHE_TTL = 3600  # 1 hour
YOY_CACHE_TTL = 1800  # 30 minutes

# Required columns for data validation
REQUIRED_COLUMNS = [
    'latitude', 
    'longitude', 
    'incident_occurred_at',
    'incident_entry_id',
    'offense_category',
    'case_status',
    'police_precinct',
    'council_district',
    'neighborhood',
    'zip_code',
    'nearest_intersection',
    'incident_day_of_week',
    'incident_hour_of_day'
]

# Categorical variables that need special formatting
CATEGORICAL_VARS = ['zip_code', 'police_precinct', 'council_district']

# Day of week labels (1=Sunday)
DAY_LABELS = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']

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