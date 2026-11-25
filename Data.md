# Detroit Crime Data Documentation

## Data Source
- **URL**: https://data.detroitmi.gov/api/download/v1/items/8e532daeec1149879bd5e67fdd9c8be0/csv?layers=0
- **Format**: CSV
- **Update Frequency**: Real-time (cached for 24 hours in application)

## Column Descriptions

| Column Name | Type | Description |
|-------------|------|-------------|
| X | Float | X coordinate (longitude) |
| Y | Float | Y coordinate (latitude) |
| incident_entry_id | String | Unique incident entry identifier |
| nearest_intersection | String | Nearest street intersection to incident |
| offense_category | String | High-level category of offense (e.g., LARCENY, AGGRAVATED ASSAULT) |
| offense_description | String | Detailed description of the offense |
| state_offense_code | String | State-specific offense code |
| arrest_charge | String | Arrest charge code |
| charge_description | String | Description of the charge |
| incident_occurred_at | DateTime | Date and time when incident occurred, published in UTC (e.g. `2019/11/30 03:33:00+00`) |
| incident_time | Time | Local (Detroit) time of incident (HH:MM:SS format) |
| incident_day_of_week | Integer | Local day of week, ISO-numbered (1=Monday ... 7=Sunday) |
| incident_hour_of_day | Integer | Local hour of day (0-23) |
| incident_year | Integer | Local year when incident occurred (the app derives its own from the converted timestamp) |
| case_id | String | Case identifier |
| case_status | String | Status of the case (ACTIVE, INACTIVE, CLEARED BY ARREST) |
| case_status_updated_at | DateTime | When case status was last updated |
| updated_in_ibr_at | DateTime | When updated in IBR system |
| updated_at | DateTime | Last update timestamp |
| crime_id | String | Crime identifier |
| report_number | String | Police report number |
| scout_car_area | String | Scout car area code |
| police_precinct | String | Police precinct number |
| census_block_2020_geoid | String | 2020 Census block group ID |
| neighborhood | String | Neighborhood name |
| council_district | String | City council district |
| zip_code | String | ZIP code |
| longitude | Float | Longitude coordinate |
| latitude | Float | Latitude coordinate |
| ESRI_OID | Integer | ESRI Object ID |

## Key Fields for Analysis

- **Location**: `latitude`, `longitude`, `nearest_intersection`, `neighborhood`
- **Time**: `incident_occurred_at`, `incident_day_of_week`, `incident_hour_of_day`, `incident_year`
- **Crime Type**: `offense_category`, `offense_description`
- **Geographic**: `police_precinct`, `council_district`, `zip_code`

## Application Data Requirements

The application requires the following columns for proper functionality:
- `latitude`, `longitude` - Geographic coordinates
- `incident_occurred_at` - Date/time information
- `offense_category`, `offense_description` - Crime classification
- `case_status` - Case status information
- `police_precinct`, `council_district`, `neighborhood` - Geographic divisions
- `zip_code`, `nearest_intersection` - Location details
- `incident_day_of_week`, `incident_hour_of_day` - Temporal analysis

Only these columns are loaded. The others, `incident_entry_id` included, are not read (counts are by row), so a change to them does not affect the app.

## Data Processing & Filtering

The application implements several data quality measures:
- **Geographic Filtering**: Latitude (42.0-43.0), Longitude (-84.0 to -82.0)
- **Local Time**: `incident_occurred_at` is converted from UTC to Detroit time (`America/Detroit`); `date` and `incident_year` are derived from the local time
- **Temporal Filtering**: Data from 2018-01-01 (local time) onwards; incidents dated after the current day are dropped
- **Unreadable Timestamps**: Rows whose timestamp cannot be read are dropped; if more than 1% are unreadable the load is refused. A timestamp that is empty or only padding is missing, not unreadable: the row is dropped and does not count towards the 1%
- **Invalid Bytes**: A byte that is not valid UTF-8 is replaced with `�` instead of failing the load
- **Text Cleanup**: Trailing padding is trimmed from category, description, status, neighborhood and intersection values; a value that is only padding is treated as missing
- **Day and Hour Numbers**: `incident_day_of_week` and `incident_hour_of_day` are read as whole numbers; a value that is not one (text, a fraction) is treated as missing
- **Quality Control**: Removes records with missing, non-numeric or out-of-range coordinates, from every figure and not only from the map; a feed left with no rows is treated as a failed load
- **Empty Export**: The portal empties the dataset while it reloads it, and an export made then is empty or has only its header. That is taken for a portal that is not ready, and the download is tried again. A file that has rows but lacks the columns the app reads is a feed that has changed: it is reported by the columns it lacks, and tried again less and less often
- **Shrunken Refresh**: A refresh left with fewer than 90% of the incidents of the copy in use is refused as incomplete (a short export, or coordinates gone missing), and the copy in use is kept. A first load has nothing to be compared with and is taken as it comes
- **Real-time Exclusion**: Excludes the feed's last day, and anything dated after it, to avoid incomplete data. The last day is the latest one with at least 10% of a typical (median) day's incidents, so a lone row with a mistyped later date is not taken for it
- **Categorical Formatting**: Zip codes, precincts and districts are read as numeric codes; rows without a valid code (a whole number above zero, written as a plain number) are left out of charts that group by them (the pivot says how many)

## Data Quality Notes

- Some records may have missing coordinates (empty X, Y values)
- The feed holds incidents reported from December 2016 to the present; an incident can have occurred before it was reported, and the app keeps those from 2018-01-01 onwards
- Real-time data with recent incidents marked as "ACTIVE"
- Each row is one offense: a report with several charges appears once per charge, so counts are offenses, not distinct reports
- The portal export typically lags the present by several days; year-to-date figures are measured up to the last day in the data. As seen on 25 November 2026: the export held incidents up to the 23rd, with about half of a usual day's on that last day (the one the app leaves out) and the days before it at about their usual level. The two weeks before it ran under a tenth below what the same weeks of earlier years would suggest, which is the most that late reports may still add to them
- Application validates data structure on load
- Comprehensive error handling for data quality issues