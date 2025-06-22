# Detroit Crime Incidents

## Main functionality

### Core Analytics (12 Interactive Tabs)
* **Categories**: Crime category analysis with drill-down capabilities
* **Case Status**: Case status distribution and tracking
* **Year over Year**: Annual trends with current year projections
* **Daily Incidents**: Historical incident count time series
* **Hour of Day of Week**: Temporal pattern heatmaps
* **Geographic Analysis**: 
  - Zip code analysis
  - Police precinct analysis
  - Council district analysis
  - Neighborhood analysis
  - Location analysis by intersection
* **Forecasting**: 1-year Prophet forecast with holiday support
* **Pivot Analysis**: Interactive pivot tables with customizable dimensions

### Advanced Features
* Real-time data filtering by offense category
* Multi-level caching system (data, forecast, year-over-year)
* Comprehensive data validation and error handling
* Holiday-aware forecasting with US holidays
* Current year projection based on year-to-date data
* Categorical data formatting for zip codes, precincts, districts

## Data sources

* Incident data to be downloaded in CSV format from https://data.detroitmi.gov/api/download/v1/items/8e532daeec1149879bd5e67fdd9c8be0/csv?layers=0
* Connect to the data source to find column names etc. Document the data in a file called Data.md
* No data stored locally, it should be downloaded when a new session is started and cached for 24 hours in memory

## Tech stack

* Python with full type hint coverage
* Streamlit for web application framework
* Pandas for data manipulation and analysis
* Plotly for interactive visualizations
* Prophet for time series forecasting with holiday support
* Modular architecture with separated constants and utilities
* Pinned dependency versions for security and stability

## Architecture

### Code Organization
* **app.py**: Main Streamlit application (12 tabs)
* **constants.py**: Configuration constants and parameters
* **utils.py**: Utility functions for data processing
* **requirements.txt**: Pinned dependencies

### Best Practices Implemented
* Type safety with comprehensive type hints
* Specific error handling patterns
* Code reusability through utility functions
* Performance optimization with multi-level caching
* Data validation and cleaning
* Categorical data formatting

## Deployment

* Docker image should use "python" as base image, no version specified
* This should be deployed using docker compose
* Multi-level caching for optimal performance
* Comprehensive error handling for production reliability
