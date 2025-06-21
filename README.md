# Detroit Crime Incidents Dashboard

An interactive web application for visualizing and analyzing Detroit crime incident data using real-time data from the Detroit Open Data Portal.

## Features

- **Interactive Map**: View crime incidents plotted on an interactive map of Detroit with detailed popups
- **Time Series Analysis**: Historical incident count trends by day
- **Heat Maps**: Crime patterns by day of week and hour of day
- **Forecasting**: 1-year crime incident forecast using Facebook Prophet
- **Category Analysis**: Top 10 crime categories visualization
- **Real-time Data**: Automatically fetches latest data from Detroit Open Data Portal
- **Filtering**: Filter data by offense category
- **Performance Optimized**: Data caching and sampling for optimal performance

## Tech Stack

- **Python**: Core programming language
- **Streamlit**: Web application framework
- **Pandas**: Data manipulation and analysis
- **Plotly**: Interactive visualizations
- **Folium**: Interactive maps
- **Prophet**: Time series forecasting
- **NumPy**: Numerical computing
- **Seaborn/Matplotlib**: Additional plotting capabilities

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
cd detroitcrime_app
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
detroitcrime_app/
├── app.py                 # Main Streamlit application
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

### Visualizations

1. **Interactive Map** (`create_map()`):
   - Folium-based map centered on Detroit
   - Circle markers for each incident
   - Popup details with offense type, location, and timestamp
   - Sampling for performance (max 1000 incidents displayed)

2. **Time Series Chart** (`create_time_series_chart()`):
   - Daily incident count over time
   - Plotly line chart with interactive features

3. **Heat Map** (`create_day_of_week_heatmap()`):
   - Day of week vs hour of day analysis
   - Shows crime patterns by time
   - Color-coded intensity visualization

4. **Forecasting** (`create_forecast()`):
   - Facebook Prophet model for 1-year predictions
   - Confidence intervals included
   - Requires minimum 30 days of historical data

5. **Category Analysis** (`create_offense_category_chart()`):
   - Top 10 crime categories by incident count
   - Horizontal bar chart for easy comparison

### Data Filtering
- Offense category filter in sidebar
- Real-time metric updates based on filters
- Responsive UI with appropriate warnings for empty data

## Performance Optimizations

- **Data Caching**: 24-hour TTL cache for API data
- **Map Sampling**: Limits map markers to 1000 for performance
- **Efficient Data Processing**: Optimized pandas operations
- **Error Handling**: Graceful handling of API failures and data issues

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

## Future Enhancements

Potential improvements could include:
- Additional crime type analyses
- Neighborhood-specific filtering
- Seasonal pattern analysis
- Crime hotspot identification
- Export functionality for data and visualizations
- Mobile-responsive design improvements

## License

[Add appropriate license information]

## Contributing

[Add contributing guidelines if applicable]