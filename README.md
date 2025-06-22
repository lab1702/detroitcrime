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
- **Filtering**: Filter all analyses by offense category
- **Performance Optimized**: Multi-level caching system for optimal performance

## Tech Stack

- **Python**: Core programming language
- **Streamlit**: Web application framework
- **Pandas**: Data manipulation and analysis
- **Plotly**: Interactive visualizations
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
├── app.py                 # Main Streamlit application (12 tabs)
├── constants.py           # Configuration constants and parameters
├── utils.py              # Utility functions for data processing
├── requirements.txt       # Python dependencies
├── Dockerfile            # Docker configuration
├── docker-compose.yml    # Docker Compose configuration
├── Data.md              # Data source documentation
├── plan.md              # Development plan
└── README.md            # This file
```

## Key Components

### Data Loading (`load_data()`)
- Fetches data from Detroit Open Data Portal
- Implements 24-hour caching for performance
- Cleans and filters data for quality
- Removes invalid coordinates and recent incomplete data

### Application Tabs

The dashboard features 12 interactive tabs:

1. **Categories**: Crime category analysis with drill-down to offense descriptions
2. **Case Status**: Distribution of case statuses across incidents
3. **Year over Year**: Annual trends with current year projections
4. **Daily Incidents**: Time series of daily incident counts
5. **Hour of Day of Week**: Temporal pattern heatmap
6. **Zip Codes**: Geographic analysis by zip code
7. **Police Precincts**: Analysis by police precinct
8. **Council Districts**: Analysis by council district
9. **Neighborhoods**: Neighborhood-level analysis
10. **Locations**: Analysis by nearest intersection
11. **Forecast**: 1-year Prophet forecast with holiday annotations
12. **Pivot Analysis**: Interactive pivot table with customizable dimensions

### Advanced Features

- **Smart Caching**: Multi-level caching (24h for data, 1h for forecast, 30min for YoY)
- **Holiday Support**: Forecast includes US holiday annotations and effects
- **Current Year Projections**: Automatic projection based on year-to-date data
- **Categorical Formatting**: Proper handling of zip codes, precincts, and districts
- **Data Validation**: Comprehensive validation and error handling
- **Type Safety**: Full type hint coverage for maintainability
- **Modular Architecture**: Separated constants, utilities, and main application

### Data Filtering
- Offense category filter in sidebar
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
- Removes records with missing coordinates
- Filters out invalid coordinates (latitude: 42.0-43.0, longitude: -84.0 to -82.0)
- Excludes the most recent date to avoid incomplete data
- Filters data from 2018-01-01 onwards for consistency

## Development Notes

- Uses latest Python packages (no version pinning)
- Follows Streamlit best practices for layout and caching
- Implements error handling for API failures
- Responsive design with wide layout configuration
- Modular code structure for maintainability

## Architecture & Code Quality

### File Structure
- **app.py**: Main Streamlit application with comprehensive type hints
- **constants.py**: Centralized configuration and magic numbers
- **utils.py**: Reusable utility functions for data processing
- **requirements.txt**: Pinned dependencies for security and stability

### Best Practices Implemented
- **Type Safety**: Full type hint coverage
- **Error Handling**: Specific exception handling patterns
- **Code Organization**: Clear separation of concerns
- **Documentation**: Comprehensive docstrings and comments
- **Performance**: Optimized caching and data processing
- **Security**: Input validation and proper dependency management

## License

[Add appropriate license information]

## Contributing

[Add contributing guidelines if applicable]