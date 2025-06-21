# Detroit Crime Incidents

## Main functionality

* View a map of incidents
* Show historical incident count by day
* View heat maps based on Day of week and Hour of Day
* Forecast incident count using prophet

## Data sources

* Incident data to be downloaded in CSV format from https://data.detroitmi.gov/api/download/v1/items/8e532daeec1149879bd5e67fdd9c8be0/csv?layers=0
* Connect to the data source to find column names etc. Document the data in a file called Data.md
* No data stored locally, it should be downloaded when a new session is started and cached for 24 hours in memory

## Tech stack

* Python
* Streamlit
* No hard coded version numbers, always use latest python packages

## Deployment

* Docker image should use "python" as base image, no version specified
* This should be deployed using docker compose
