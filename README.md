# Detroit Crime Incidents Dashboard

An interactive web application for visualizing and analyzing Detroit crime incident data using real-time data from the Detroit Open Data Portal.

## Features

- **Crime Categories**: Top 10 crime categories and offense descriptions analysis
- **Case Status Analysis**: Distribution of case statuses (Active, Inactive, Cleared by Arrest)
- **Year over Year Analysis**: Annual incident trends with current year projections
- **Time Series Analysis**: Historical incident count trends by day
- **Temporal Heat Maps**: Crime patterns by day of week and hour of day
- **Geographic Analysis**: 
  - Top 10 zip codes by incident count
  - Police precinct analysis
  - Council district analysis
  - Neighborhood analysis
  - Location analysis by nearest intersection
- **Forecasting**: 1-year crime incident forecast using Facebook Prophet with holiday support
- **Pivot Analysis**: Interactive pivot table analysis with customizable rows/columns
- **Real-time Data**: Automatically fetches latest data from Detroit Open Data Portal
- **Map**: Scatter and density views of incident locations
- **Filtering**: Filter all analyses by offense category and year
- **Performance Optimized**: Multi-level caching system for optimal performance

## Tech Stack

- **Python**: Core programming language
- **Streamlit**: Web application framework
- **Pandas**: Data manipulation and analysis
- **Plotly**: Interactive visualizations
- **PyDeck**: Map rendering (scatter and heatmap layers)
- **Prophet**: Time series forecasting with holiday support
- **NumPy**: Numerical computing
- **Requests**: HTTP library for data fetching

## Data Source

The application fetches real-time crime incident data from the Detroit Open Data Portal:
- **API Endpoint**: `https://data.detroitmi.gov/api/download/v1/items/8e532daeec1149879bd5e67fdd9c8be0/csv`
- **Format**: CSV
- **Update Frequency**: Real-time (cached for 24 hours in application)
- **Coverage**: Crime incidents from 2018 onwards
- **Coordinates**: Filtered for valid Detroit area coordinates
- **Time zone**: Feed timestamps are UTC and are converted to Detroit local time on load

## Installation & Setup

### Prerequisites
- Python 3.12 or higher
- Docker (for containerized deployment)

### Local Development

1. Clone the repository:
```bash
git clone <repository-url>
cd detroitcrime
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Run the application:
```bash
streamlit run app.py
```

The application will be available at `http://localhost:8501`

### Docker Deployment

1. Build and run using Docker Compose:
```bash
docker-compose up --build
```

The application will be available at `http://localhost:8502`

## Application Structure

```
detroitcrime/
├── app.py                 # Entry point: data loading, sidebar filters, Overview page
├── pages/
│   ├── 2_Trends.py        # Daily trend, year over year, hour x day heatmap, forecast
│   ├── 3_Geography.py     # Neighborhoods, precincts, districts, zip codes, locations
│   ├── 4_Analysis.py      # Interactive pivot table
│   └── 5_Map.py           # Scatter and heatmap map
├── constants.py           # Configuration constants and parameters
├── utils.py               # Data loading, cleaning and year-over-year helpers
├── theme.py               # Dark theme CSS and Plotly styling
├── tests/                 # pytest suite (unit tests and headless page tests)
├── requirements.txt       # Python dependencies
├── requirements-dev.txt   # Adds test dependencies
├── Dockerfile             # Docker configuration
├── docker-compose.yml     # Docker Compose configuration
├── Data.md                # Data source documentation
├── docs/plans/            # Design notes
└── README.md              # This file
```

## Key Components

### Data Loading (`fetch_data()` and `dataset_cache()`)
- Fetches data from Detroit Open Data Portal, retrying network errors, rate limiting and server errors
- Keeps the data for 24 hours, then refreshes it; if a refresh fails, the previous copy stays in use and a warning is shown
- Downloads on a background thread shared by every session: visitors who arrive during a download wait for that one, and a click while waiting does not restart it
- A failed first load is reported to every visitor and tried again after a minute, rather than repeated for each of them
- Cleans and filters data for quality
- Removes invalid coordinates and recent incomplete data

### Application Pages

The dashboard has five pages, all driven by the sidebar filters:

1. **Overview**: KPI cards with year-over-year change (shown for all years and for the current year), top categories (or offense descriptions when a category is selected), case status, daily incidents. The daily average is measured over the whole selected period, including days without incidents
2. **Trends**: Daily incidents, year over year with a projection for the latest year, hour of day x day of week heatmap, 1-year Prophet forecast with holiday annotations. The forecast is fitted on every year of the selected category, so the Year filter does not apply to it
3. **Geography**: Top neighborhoods, police precincts, council districts, zip codes and intersections
4. **Pivot Analysis**: Interactive pivot table with customizable rows and columns. Incidents with no value for a chosen field cannot be placed in the table; a note under it says how many
5. **Map**: Incident locations as colour-coded points (sampled above 250,000 points) or as a density heatmap

### Advanced Features

- **Smart Caching**: Multi-level caching (24h for data, 1h for forecast, 30min for YoY)
- **Holiday Support**: Forecast includes US holiday annotations and effects
- **Current Year Projections**: Automatic projection based on year-to-date data
- **Categorical Formatting**: Proper handling of zip codes, precincts, and districts
- **Data Validation**: Comprehensive validation and error handling
- **Type Safety**: Full type hint coverage for maintainability
- **Modular Architecture**: Separated constants, utilities, and main application

### Data Filtering
- Offense category and year filters in sidebar
- Real-time metric updates based on filters
- Responsive UI with appropriate warnings for empty data

## Performance Optimizations

- **Multi-level Caching**: Different TTL values for different data types
  - Data loading: 24 hours
  - Forecasting: 1 hour
  - Year-over-year analysis: 30 minutes
- **Efficient Data Processing**: Utility functions for common operations
- **Categorical Optimization**: Specialized handling for geographic identifiers
- **Error Handling**: Specific exception handling with user-friendly messages
- **Memory Management**: Optimized DataFrame operations and copying

## Environment Variables

The application uses the following environment variables (set in docker-compose.yml):
- `PYTHONUNBUFFERED=1`: Ensures Python output is not buffered

## Data Quality & Filtering

The application implements several data quality measures:
- Removes records with missing or non-numeric coordinates
- Filters out invalid coordinates (latitude: 42.0-43.0, longitude: -84.0 to -82.0)
- Converts feed timestamps from UTC to Detroit local time before deriving dates and years
- Drops timestamps that cannot be read, and refuses the load if more than 1% of them are unreadable
- Drops incidents dated after the current day
- Excludes the feed's last day, and anything dated after it, to avoid incomplete data. The last day is the latest one with at least 10% of a typical day's incidents, so a lone mistyped date is not taken for it
- Filters data from 2018-01-01 onwards for consistency
- Trims padded text values so the same value is not counted twice, and treats a value that is only padding as missing
- Replaces bytes that are not valid UTF-8 instead of refusing the file

## Development Notes

- Dependencies are pinned to tested version ranges in `requirements.txt`; the Docker image uses `python:3.15`
- Run the tests with `pip install -r requirements-dev.txt` then `pytest`
- Follows Streamlit best practices for layout and caching
- Implements error handling for API failures
- Responsive design with wide layout configuration
- Modular code structure for maintainability

## Architecture & Code Quality

### File Structure
- **app.py**: Main Streamlit application with comprehensive type hints
- **constants.py**: Centralized configuration and magic numbers
- **utils.py**: Reusable utility functions for data processing
- **requirements.txt**: Dependencies pinned to tested version ranges

### Best Practices Implemented
- **Type Safety**: Full type hint coverage
- **Error Handling**: Specific exception handling patterns
- **Code Organization**: Clear separation of concerns
- **Documentation**: Comprehensive docstrings and comments
- **Performance**: Optimized caching and data processing
- **Security**: Input validation and proper dependency management

## License

MIT License. See [LICENSE](LICENSE).