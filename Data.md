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
| incident_occurred_at | DateTime | Date and time when incident occurred |
| incident_time | Time | Time of incident (HH:MM:SS format) |
| incident_day_of_week | Integer | Day of week (1-7, where 1=Sunday) |
| incident_hour_of_day | Integer | Hour of day (0-23) |
| incident_year | Integer | Year when incident occurred |
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

## Data Quality Notes

- Some records may have missing coordinates (empty X, Y values)
- Incidents span multiple years (2017-2025 based on sample)
- Real-time data with recent incidents marked as "ACTIVE"