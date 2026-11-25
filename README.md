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
docker compose up --build
```
(With the older standalone Compose, the command is `docker-compose up --build`.)

The application will be available at `http://localhost:8502`

## Application Structure

```
detroitcrime/
├── app.py                 # Entry point: data loading, sidebar filters, Overview page
├── views/                 # Page scripts, run by app.py (not named pages/: see Development Notes)
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
- Fetches data from Detroit Open Data Portal, retrying network errors, rate limiting, server errors and an export that arrives empty or with only its header (the portal empties the dataset while it reloads it)
- Gives up on a download attempt when the portal has said nothing for a minute (whether before its first byte or in the middle of the download, it is reported as a portal that took too long), and starts no further attempt once two minutes have passed since the first, so a portal that has stalled is reported within a few minutes rather than waited for behind a spinner; a download that is still coming in is not cut off by either. One attempt is given up, as a portal that took too long, once it has run for half an hour (the export usually arrives in seconds): a download that only trickles in would otherwise never end, and every later load waits behind the one under way
- Keeps the data for 24 hours (by the wall clock, so the hours a sleeping host spends suspended count), then refreshes it in the background: nobody waits for a refresh, and the previous copy is shown until the new one has arrived. A page that is open while a refresh runs says so, and draws itself again when the refresh has ended (it asks every 5 seconds), so a visitor who only looks once is not left with the old figures; that includes a refresh that ends just as the page is being drawn. A page that is left open with nothing to wait for looks again once a minute: it follows a refresh that another visitor set off, and sets the refresh off itself once the data is 24 hours old (or a failed refresh is due another try), so a dashboard left open does not keep the figures it was first drawn with. If a refresh fails, the previous copy stays in use and a warning is shown
- Refuses a refresh that has fewer than 99% of the incidents of the copy in use (a short export, or one that has lost its coordinates), or that ends more than 3 days before it (an export cut short by date, which can keep nearly all of its rows): the copy in use stays, with the warning. If the feed really has shrunk that much, restart the app to accept it (the server log says so; the warning on the page asks nothing of visitors)
- Refuses any load, the first included, whose incidents end more than 30 days before today: the export runs a few days behind, and one that ends a month back has been cut short or is no longer updated, so the year so far would be measured up to a day long past. A first load has no earlier copy to be held against, so this is the one check on it; it is reported as an error and tried again less and less often, like any failure that would only repeat. If the portal really has stopped updating the feed, raise `MAX_FEED_LAG_DAYS` in `constants.py` to show it as it is (the server log says so)
- Tries a failed refresh again after 15 minutes. A failure that would only repeat (a refused refresh, a feed that ends long ago, a feed that has lost a column, a 404) costs a whole download each time, so it waits twice as long after each one in a row: 15 minutes, 30, an hour, up to six hours (not longer: a short export made while the portal reloads its data looks like a feed that has shrunk, and does pass). A failed first load is spaced out the same way, starting from its one minute
- Downloads on a background thread shared by every session: visitors who arrive during the first download wait for that one, and a click while waiting does not restart it
- A failed first load is reported to every visitor and tried again after a minute, rather than repeated for each of them. When the failure is one that can pass (network trouble, rate limiting, a server error, a portal that is not ready), an open page goes back for the data by itself once that minute is up (it is counted from the failure, whenever the page was opened). After one that would only repeat, such as a 404 or a feed that cannot be read, an open page goes back by itself as well, but no sooner than the lengthening wait above allows (it looks once a minute): what looks set to repeat can pass after all, and a screen left open is not left on a failure that is over
- A load that fails is written to the server log with its traceback; the pages show only the message. The same goes for a forecast or a pivot table that fails
- Cleans and filters data for quality
- Removes invalid coordinates and recent incomplete data

### Application Pages

The dashboard has five pages, all driven by the sidebar filters:

1. **Overview**: KPI cards with year-over-year change (shown for all years and for the current year; it compares 1 January to the last day of data with the same days of last year, and is left out unless the data reaches back to 1 January of last year; a change that rounds to zero is shown in grey, as neither a rise nor a fall), top categories (or offense descriptions when a category is selected), case status, daily incidents. The daily average is measured over the whole selected period, including days without incidents. The card that names a category is set smaller than the number cards and wraps, so a long name is not cut off
2. **Trends**: Daily incidents, year over year with a projection for the latest year, hour of day x day of week heatmap, 1-year Prophet forecast with holiday annotations. Incidents whose day or hour is missing or out of range have no cell in the heatmap; a note under it says how many. Each holiday is marked once on the forecast, on the day itself and not also on the day it is observed, under a name short enough to read ("Juneteenth", "MLK Day"); one that falls on the last day of data is history and is not marked. A year the data covers only in part (the current one, and the first if the data does not start on 1 January) is drawn lighter, with a note saying which part; the projection is scaled from the days of the year the data covers and needs 60 of them. The forecast is fitted on every year of the selected category, so the Year filter does not apply to it; a selection with incidents on fewer than 30 days is not forecast
3. **Geography**: Top neighborhoods, zip codes and intersections, and every police precinct and council district
4. **Pivot Analysis**: Interactive pivot table with customizable rows and columns. Incidents with no value for a chosen field cannot be placed in the table; a note under it says how many
5. **Map**: Incident locations as colour-coded points (sampled above 50,000 points) or as a density heatmap, which always uses every incident (grouped into coarser spots if there are more than 100,000 of them to send). The 15 most frequent categories have a colour each; the rest, and incidents with no category, share a grey that the legend names. The chosen view is kept while other pages are visited

### Advanced Features

- **Smart Caching**: Multi-level caching (24h for data, 1h for forecast, 30min for YoY)
- **Holiday Support**: Forecast includes US holiday annotations and effects
- **Current Year Projections**: Automatic projection based on year-to-date data
- **Categorical Formatting**: Proper handling of zip codes, precincts, and districts
- **Data Validation**: Comprehensive validation and error handling
- **Type Safety**: Full type hint coverage for maintainability
- **Modular Architecture**: Separated constants, utilities, and main application

### Data Filtering
- Offense category and year filters in sidebar; a choice is kept when a refresh changes the list of options, and falls back to "All" only if it is no longer offered
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
- Filters out invalid coordinates (latitude: 42.0-43.0, longitude: -84.0 to -82.0). Records removed for their coordinates are left out of every figure, not only the map; the Total Incidents help says so
- Converts feed timestamps from UTC to Detroit local time before deriving dates and years
- Drops timestamps that cannot be read, and refuses the load if more than 1% of them are unreadable. A blank timestamp is a missing one, not an unreadable one
- Drops incidents dated after the current day
- Refuses a feed whose incidents end more than 30 days before the current day, as cut short or no longer updated
- Excludes the feed's last day, and anything dated after it, to avoid incomplete data. The last day is the latest one with at least 10% of a typical day's incidents, so a lone mistyped date is not taken for it
- Filters data from 2018-01-01 onwards for consistency
- Trims padded text values so the same value is not counted twice, and treats a value that is only padding as missing
- Reads the feed's day-of-week and hour-of-day fields as whole numbers, and treats a value that is not one as missing
- Reads precinct, district and zip codes as whole numbers once, when the data is loaded, and treats a value that is not a valid code as missing
- Replaces bytes that are not valid UTF-8 instead of refusing the file

## Development Notes

- Dependencies are pinned to tested version ranges in `requirements.txt`; the Docker image uses `python:3.15`
- Run the tests with `pip install -r requirements-dev.txt` then `pytest`
- The page scripts are in `views/` and must not be moved to a folder named `pages/`. Streamlit runs the scripts of a `pages/` folder on their own until `st.navigation` has been called once, so the first visitor to a new server who arrived on a page's URL would get that page without the data `app.py` loads
- Each page script draws itself inside a `render()` function, as the Overview does in `overview_page()`, and keeps the data in that function's variables. At the top level of the script the data would stay in the script's namespace after the run: Streamlit keeps a button's callback, and so the namespace it was defined in, for as long as the tab is open, and any function defined in the script keeps the namespace until Python next collects unreachable objects. Either way an idle tab would hold a copy of its selection in memory
- The Docker image runs the app as an unprivileged user (`app`), which can read the app's files but not change them. The Dockerfile sets the file modes itself, because files copied from a Windows checkout (a build under WSL) would otherwise be writable by everyone
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