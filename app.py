"""Detroit Crime Incidents Dashboard

A Streamlit application for visualizing and analyzing Detroit crime incident data.
Provides interactive charts, forecasting, and pivot analysis capabilities.
"""

from typing import Optional, Tuple, Any
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from prophet import Prophet
from prophet.make_holidays import make_holidays_df
from datetime import datetime
import requests

from constants import (
    DATA_URL, DATA_CACHE_TTL, CHART_HEIGHT, TOP_N_ITEMS, MIN_FORECAST_DAYS, 
    FORECAST_PERIODS, HISTORICAL_MONTHS, MIN_PROJECTION_DAYS,
    DAY_LABELS, COLOR_SCALE_HEATMAP, PIVOT_COLUMNS, FORECAST_CACHE_TTL, YOY_CACHE_TTL,
    MIN_DATE, DETROIT_LAT_MIN, DETROIT_LAT_MAX, DETROIT_LON_MIN, DETROIT_LON_MAX,
    CATEGORICAL_VARS
)
from utils import (
    validate_data, format_categorical_column, format_zip_code_column, 
    safe_load_data_from_url, clean_and_filter_data
)

# Configure page
st.set_page_config(
    page_title="Detroit Crime Incidents",
    page_icon="🚔",
    layout="wide"
)

# Cache for 24 hours
@st.cache_data(ttl=DATA_CACHE_TTL, show_spinner=False)
def load_data() -> Optional[pd.DataFrame]:
    """Load Detroit crime data with improved error handling and validation"""
    try:
        # Load data with proper error handling
        df = safe_load_data_from_url(DATA_URL)
        
        # Validate data structure
        validate_data(df)
        
        # Clean and filter data
        lat_bounds = (DETROIT_LAT_MIN, DETROIT_LAT_MAX)
        lon_bounds = (DETROIT_LON_MIN, DETROIT_LON_MAX)
        df = clean_and_filter_data(df, MIN_DATE, lat_bounds, lon_bounds)
        
        return df
        
    except requests.exceptions.Timeout:
        st.error("⏱️ Request timed out. The data source may be slow. Please try again.")
        return None
    except requests.exceptions.ConnectionError:
        st.error("🌐 Connection failed. Please check your internet connection and try again.")
        return None
    except requests.exceptions.HTTPError as e:
        st.error(f"🚫 Server error: {str(e)}. The data source may be temporarily unavailable.")
        return None
    except ValueError as e:
        st.error(f"📊 Data validation error: {str(e)}")
        return None
    except Exception as e:
        st.error(f"❌ Unexpected error loading data: {str(e)}")
        return None



def create_time_series_chart(df):
    """Create historical incident count by day"""
    daily_counts = df.groupby('date').size().reset_index(name='count')
    daily_counts['date'] = pd.to_datetime(daily_counts['date'])
    
    fig = px.line(daily_counts, x='date', y='count', 
                  labels={'date': 'Date', 'count': 'Number of Incidents'})
    fig.update_layout(height=CHART_HEIGHT)
    return fig

@st.cache_data(ttl=3600)
def create_day_of_week_heatmap(df):
    """Create heatmap for day of week vs hour of day"""
    # Create pivot table
    heatmap_data = df.groupby(['incident_day_of_week', 'incident_hour_of_day']).size().reset_index(name='count')
    pivot_data = heatmap_data.pivot(index='incident_day_of_week', columns='incident_hour_of_day', values='count')
    pivot_data = pivot_data.fillna(0)
    
    # Day of week labels (1=Sunday)
    day_labels = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']
    
    fig = px.imshow(pivot_data, 
                    labels=dict(x="Hour of Day", y="Day of Week", color="Incident Count"),
                    aspect="auto",
                    color_continuous_scale="RdYlGn_r")
    
    fig.update_yaxes(tickvals=list(range(1, 8)), ticktext=day_labels)
    fig.update_layout(height=CHART_HEIGHT)
    return fig

@st.cache_data(ttl=FORECAST_CACHE_TTL)
def create_forecast(df: pd.DataFrame) -> Optional[go.Figure]:
    """Create Prophet forecast for incident count"""
    try:
        # Prepare data for Prophet
        daily_counts = df.groupby('date').size().reset_index()
        daily_counts.columns = ['ds', 'y']
        daily_counts['ds'] = pd.to_datetime(daily_counts['ds'])
        
        # Ensure we have enough data points
        if len(daily_counts) < 30:
            st.warning("Not enough historical data for reliable forecasting (minimum 30 days required)")
            return None
        
        # Create holidays dataframe for US holidays
        holidays = make_holidays_df(year_list=list(range(daily_counts['ds'].dt.year.min(), 
                                                        daily_counts['ds'].dt.year.max() + 2)), 
                                  country='US')
        
        # Create and fit model with holidays
        model = Prophet(holidays=holidays)
        model.fit(daily_counts)
        
        # Make future dataframe for 1 year (365 days)
        future = model.make_future_dataframe(periods=365)
        forecast = model.predict(future)
        
        # Create forecast plot
        fig = go.Figure()
        
        # Historical data - only show last 3 months before forecast
        forecast_start = daily_counts['ds'].max()
        three_months_before = forecast_start - pd.DateOffset(months=3)
        historical_data = daily_counts[daily_counts['ds'] >= three_months_before]
        
        fig.add_trace(go.Scatter(
            x=historical_data['ds'], 
            y=historical_data['y'],
            mode='markers',
            name='Historical Data',
            marker=dict(color='blue', size=4)
        ))
        
        # Forecast
        future_dates = forecast[forecast['ds'] > daily_counts['ds'].max()]
        fig.add_trace(go.Scatter(
            x=future_dates['ds'],
            y=future_dates['yhat'],
            mode='lines',
            name='Forecast',
            line=dict(color='red', width=2)
        ))
        
        # Confidence interval
        fig.add_trace(go.Scatter(
            x=future_dates['ds'],
            y=future_dates['yhat_upper'],
            fill=None,
            mode='lines',
            line_color='rgba(0,0,0,0)',
            showlegend=False
        ))
        
        fig.add_trace(go.Scatter(
            x=future_dates['ds'],
            y=future_dates['yhat_lower'],
            fill='tonexty',
            mode='lines',
            line_color='rgba(0,0,0,0)',
            name='Confidence Interval',
            fillcolor='rgba(255,0,0,0.2)'
        ))
        
        # Add holiday annotations
        # Filter holidays to show only those in the forecast period
        forecast_start = daily_counts['ds'].max()
        forecast_end = future_dates['ds'].max()
        forecast_holidays = holidays[
            (holidays['ds'] >= forecast_start) & 
            (holidays['ds'] <= forecast_end)
        ]
        
        # Add vertical lines and annotations for major holidays
        if not forecast_holidays.empty:
            y_max = future_dates['yhat'].max()
            y_min = future_dates['yhat'].min()
            y_range = y_max - y_min
            
            # Sort holidays by date to better handle spacing
            forecast_holidays = forecast_holidays.sort_values('ds')
            
            # Use simple alternating levels to prevent overlaps
            for i, (_, holiday) in enumerate(forecast_holidays.iterrows()):
                holiday_date = holiday['ds']
                holiday_name = holiday['holiday']
                
                # Add vertical line for holiday
                fig.add_vline(
                    x=holiday_date,
                    line_dash="dash",
                    line_color="rgba(100, 200, 100, 0.6)",
                    line_width=1
                )
                
                # Simple alternating levels - each holiday gets a different level
                level = i % 4  # Rotate through 4 levels
                
                # Calculate position based on level
                y_position = y_max - (y_range * 0.1) - level * (y_range * 0.25)
                arrow_direction = -25 - level * 25
                
                # Truncate long holiday names
                display_name = holiday_name if len(holiday_name) <= 12 else holiday_name[:9] + "..."
                
                # Add annotation for holiday name with better styling
                fig.add_annotation(
                    x=holiday_date,
                    y=y_position,
                    text=display_name,
                    showarrow=True,
                    arrowhead=1,
                    arrowsize=0.7,
                    arrowwidth=1,
                    arrowcolor="rgba(100, 200, 100, 0.8)",
                    ax=0,
                    ay=arrow_direction,
                    bgcolor="rgba(50, 50, 50, 0.9)",
                    bordercolor="rgba(100, 200, 100, 0.8)",
                    borderwidth=1,
                    font=dict(size=8, color="white")
                )
        
        fig.update_layout(
            xaxis_title='Date',
            yaxis_title='Incident Count',
            height=CHART_HEIGHT
        )
        
        return fig
    except ValueError as e:
        st.error(f"Data validation error in forecast: {str(e)}")
        return None
    except Exception as e:
        st.error(f"Unexpected error creating forecast: {str(e)}")
        return None

@st.cache_data(ttl=1800)
def create_year_over_year_chart(df):
    """Create year over year chart showing incident counts by year with current year projection"""
    from datetime import datetime
    
    df['year'] = df['incident_occurred_at'].dt.year
    yearly_counts = df.groupby('year').size().reset_index(name='count')
    
    # Get current year and check if we have enough data for projection
    current_year = datetime.now().year
    current_year_data = df[df['year'] == current_year]
    
    # Create base bar chart
    fig = px.bar(yearly_counts, x='year', y='count',
                 labels={'year': 'Year', 'count': 'Number of Incidents'})
    
    # Add projection for current year if we have at least 2 months of data
    if len(current_year_data) > 0:
        # Calculate days elapsed in current year
        current_date = datetime.now()
        start_of_year = datetime(current_year, 1, 1)
        days_elapsed = (current_date - start_of_year).days
        
        # Check if we have at least minimum data for projection
        if days_elapsed >= MIN_PROJECTION_DAYS:
            # Calculate projection
            incidents_so_far = len(current_year_data)
            days_in_year = 366 if current_year % 4 == 0 else 365  # Account for leap year
            projected_total = incidents_so_far * (days_in_year / days_elapsed)
            
            # Add transparent bar for projection (above current year's actual count)
            current_count = yearly_counts[yearly_counts['year'] == current_year]['count'].iloc[0] if current_year in yearly_counts['year'].values else 0
            projection_height = projected_total - current_count
            
            if projection_height > 0:
                fig.add_bar(
                    x=[current_year],
                    y=[projection_height],
                    base=[current_count],
                    name='Projected (End of Year)',
                    marker_color='rgba(255, 165, 0, 0.3)',
                    showlegend=True
                )
    
    fig.update_layout(height=CHART_HEIGHT, barmode='overlay')
    return fig

@st.cache_data(ttl=3600)
def create_offense_category_chart(df):
    """Create chart showing distribution of offense categories"""
    offense_counts = df['offense_category'].value_counts().head(10)
    
    fig = px.bar(
        x=offense_counts.values[::-1],
        y=offense_counts.index[::-1],
        orientation='h',
        labels={'x': 'Number of Incidents', 'y': 'Offense Category'}
    )
    fig.update_layout(height=CHART_HEIGHT)
    return fig

def create_offense_description_chart(df):
    """Create chart showing distribution of offense descriptions for filtered category"""
    offense_counts = df['offense_description'].value_counts().head(10)
    
    fig = px.bar(
        x=offense_counts.values[::-1],
        y=offense_counts.index[::-1],
        orientation='h',
        labels={'x': 'Number of Incidents', 'y': 'Offense Description'}
    )
    fig.update_layout(height=CHART_HEIGHT)
    return fig

@st.cache_data(ttl=3600)
def create_case_status_chart(df):
    """Create chart showing all case statuses by incident count"""
    status_counts = df['case_status'].value_counts()
    
    fig = px.bar(
        x=status_counts.values[::-1],
        y=status_counts.index[::-1],
        orientation='h',
        labels={'x': 'Number of Incidents', 'y': 'Case Status'}
    )
    fig.update_layout(height=CHART_HEIGHT)
    return fig

def create_pivot_chart(df, rows, columns, values, aggfunc):
    """Create pivot table and chart"""
    try:
        # Only work with required columns to save memory
        required_cols = list(set([rows, columns, values]))
        df_subset = df[required_cols].copy()
        
        # Format only the categorical variables that are actually used
        if rows in CATEGORICAL_VARS:
            prefix = 'Precinct' if rows == 'police_precinct' else 'District' if rows == 'council_district' else ''
            if rows == 'zip_code':
                df_subset = format_zip_code_column(df_subset, rows)
            else:
                df_subset = format_categorical_column(df_subset, rows, prefix)
                
        if columns in CATEGORICAL_VARS and columns != rows:  # Avoid double processing
            prefix = 'Precinct' if columns == 'police_precinct' else 'District' if columns == 'council_district' else ''
            if columns == 'zip_code':
                df_subset = format_zip_code_column(df_subset, columns)
            else:
                df_subset = format_categorical_column(df_subset, columns, prefix)
        
        # Create pivot table with optimized data
        pivot_table = pd.pivot_table(
            df_subset, 
            index=rows, 
            columns=columns, 
            values=values, 
            aggfunc=aggfunc, 
            fill_value=0
        )
        
        # Force categorical formatting on pivot table index and columns
        if rows == 'zip_code':
            pivot_table.index = pivot_table.index.astype(str)
        if columns == 'zip_code':
            pivot_table.columns = pivot_table.columns.astype(str)
        
        # Create heatmap for pivot visualization
        fig = px.imshow(
            pivot_table,
            labels=dict(x=columns, y=rows, color="Count"),
            aspect="auto",
            color_continuous_scale=COLOR_SCALE_HEATMAP
        )
        
        # Force categorical axis formatting in plotly
        if rows == 'zip_code':
            fig.update_yaxes(type='category')
        if columns == 'zip_code':
            fig.update_xaxes(type='category')
        fig.update_layout(height=CHART_HEIGHT)
        
        return pivot_table, fig
    except (ValueError, KeyError) as e:
        st.error(f"Data processing error in pivot table: {str(e)}")
        return None, None
    except Exception as e:
        st.error(f"Unexpected error creating pivot table: {str(e)}")
        return None, None

def create_zip_code_chart(df):
    """Create chart showing top 10 zip codes by incident count"""
    # Ensure zip_code is treated as categorical string, removing decimals
    df_copy = df.copy()
    df_copy['zip_code'] = pd.to_numeric(df_copy['zip_code'], errors='coerce').fillna(0).astype(int).astype(str)
    # Remove '0' entries that came from NaN values
    df_copy = df_copy[df_copy['zip_code'] != '0']
    zip_counts = df_copy['zip_code'].value_counts().head(10)
    
    fig = px.bar(
        x=zip_counts.values[::-1],
        y=[str(idx) for idx in zip_counts.index[::-1]],
        orientation='h',
        labels={'x': 'Number of Incidents', 'y': 'Zip Code'}
    )
    fig.update_layout(height=CHART_HEIGHT)
    fig.update_yaxes(type='category')
    return fig

def create_police_precinct_chart(df):
    """Create chart showing top N police precincts by incident count"""
    df_formatted = format_categorical_column(df, 'police_precinct', 'Precinct')
    precinct_counts = df_formatted['police_precinct'].value_counts().head(TOP_N_ITEMS)
    
    fig = px.bar(
        x=precinct_counts.values[::-1],
        y=precinct_counts.index[::-1],
        orientation='h',
        labels={'x': 'Number of Incidents', 'y': 'Police Precinct'}
    )
    fig.update_layout(height=CHART_HEIGHT)
    fig.update_yaxes(type='category')
    return fig

def create_council_district_chart(df):
    """Create chart showing top N council districts by incident count"""
    df_formatted = format_categorical_column(df, 'council_district', 'District')
    district_counts = df_formatted['council_district'].value_counts().head(TOP_N_ITEMS)
    
    fig = px.bar(
        x=district_counts.values[::-1],
        y=district_counts.index[::-1],
        orientation='h',
        labels={'x': 'Number of Incidents', 'y': 'Council District'}
    )
    fig.update_layout(height=CHART_HEIGHT)
    fig.update_yaxes(type='category')
    return fig

def create_neighborhood_chart(df):
    """Create chart showing top N neighborhoods by incident count"""
    neighborhood_counts = df['neighborhood'].value_counts().head(TOP_N_ITEMS)
    
    fig = px.bar(
        x=neighborhood_counts.values[::-1],
        y=neighborhood_counts.index[::-1],
        orientation='h',
        labels={'x': 'Number of Incidents', 'y': 'Neighborhood'}
    )
    fig.update_layout(height=CHART_HEIGHT)
    return fig

def create_location_chart(df):
    """Create chart showing top N nearest intersections by incident count"""
    location_counts = df['nearest_intersection'].value_counts().head(TOP_N_ITEMS)
    
    fig = px.bar(
        x=location_counts.values[::-1],
        y=location_counts.index[::-1],
        orientation='h',
        labels={'x': 'Number of Incidents', 'y': 'Nearest Intersection'}
    )
    fig.update_layout(height=CHART_HEIGHT)
    return fig

def render_chart_tab(df_filtered: pd.DataFrame, chart_func, title: str, **kwargs) -> None:
    """Helper function to render a chart tab with consistent error handling
    
    Args:
        df_filtered: Filtered DataFrame
        chart_func: Function to create the chart
        title: Chart title/subtitle
        **kwargs: Additional arguments to pass to chart function
    """
    st.subheader(title)
    if not df_filtered.empty:
        try:
            fig = chart_func(df_filtered, **kwargs)
            if fig is not None:
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.error("Unable to create chart. Please try different parameters.")
        except Exception as e:
            st.error(f"Error creating chart: {str(e)}")
    else:
        st.warning("No data available for selected filters")


@st.cache_data(ttl=1800)
def get_filtered_data(df: pd.DataFrame, selected_category: str) -> pd.DataFrame:
    """Get filtered DataFrame with caching to prevent recomputation
    
    Args:
        df: Source DataFrame
        selected_category: Selected offense category or 'All'
        
    Returns:
        Filtered DataFrame
    """
    if selected_category == 'All':
        return df
    return df[df['offense_category'] == selected_category]


def main() -> None:
    """Main application function that sets up the Streamlit dashboard"""
    st.title("🚔 Detroit Crime Incidents Dashboard")
    st.markdown("Interactive visualization of Detroit crime incident data")
    
    # Load data
    df = load_data()
    if df is None:
        st.error("Failed to load data. Please try again later.")
        return
    
    # Sidebar filters
    st.sidebar.header("Filters")
    
    # Offense category filter  
    offense_categories = ['All'] + sorted(df['offense_category'].unique().tolist())
    selected_category = st.sidebar.selectbox("Offense Category", offense_categories)
    
    # Filter data with caching
    df_filtered = get_filtered_data(df, selected_category)
    
    # Display metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Incidents", len(df_filtered))
    with col2:
        min_date = df_filtered['date'].min()
        max_date = df_filtered['date'].max()
        total_days = (max_date - min_date).days + 1
        st.metric("Date Range", f"{total_days} days", f"{min_date} to {max_date}")
    with col3:
        avg_daily = len(df_filtered) / total_days if total_days > 0 else 0
        st.metric("Avg Daily", f"{avg_daily:.1f}")
    with col4:
        if selected_category != 'All':
            st.metric("Selected Category", selected_category)
        else:
            most_common = df_filtered['offense_category'].mode()[0] if not df_filtered.empty else "N/A"
            st.metric("Most Common", most_common)
    
    # Main content tabs
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10, tab11, tab12 = st.tabs(["Categories", "Case Status", "Year over Year", "Daily Incidents", "Hour of Day of Week", "Zip Codes", "Police Precincts", "Council Districts", "Neighborhoods", "Locations", "Forecast", "Pivot Analysis"])
    
    with tab1:
        if selected_category != 'All':
            st.subheader(f"Top {TOP_N_ITEMS} Offense Descriptions - {selected_category}")
            if not df_filtered.empty:
                description_fig = create_offense_description_chart(df_filtered)
                st.plotly_chart(description_fig, use_container_width=True)
            else:
                st.warning("No data available for selected filters")
        else:
            st.subheader(f"Top {TOP_N_ITEMS} Crime Categories")
            if not df_filtered.empty:
                category_fig = create_offense_category_chart(df_filtered)
                st.plotly_chart(category_fig, use_container_width=True)
            else:
                st.warning("No data available for selected filters")
    
    with tab2:
        st.subheader("Case Status Distribution")
        if not df_filtered.empty:
            case_status_fig = create_case_status_chart(df_filtered)
            st.plotly_chart(case_status_fig, use_container_width=True)
        else:
            st.warning("No data available for selected filters")
    
    with tab3:
        st.subheader("Year over Year Incident Count")
        if not df_filtered.empty:
            yoy_fig = create_year_over_year_chart(df_filtered)
            st.plotly_chart(yoy_fig, use_container_width=True)
        else:
            st.warning("No data available for selected filters")
    
    with tab4:
        st.subheader("Historical Incident Count")
        if not df_filtered.empty:
            time_series_fig = create_time_series_chart(df_filtered)
            st.plotly_chart(time_series_fig, use_container_width=True)
        else:
            st.warning("No data available for selected filters")
    
    with tab5:
        st.subheader("Day of Week vs Hour of Day Heatmap")
        if not df_filtered.empty:
            heatmap_fig = create_day_of_week_heatmap(df_filtered)
            st.plotly_chart(heatmap_fig, use_container_width=True)
        else:
            st.warning("No data available for selected filters")
    
    with tab6:
        st.subheader(f"Top {TOP_N_ITEMS} Zip Codes by Incident Count")
        if not df_filtered.empty:
            zip_fig = create_zip_code_chart(df_filtered)
            st.plotly_chart(zip_fig, use_container_width=True)
        else:
            st.warning("No data available for selected filters")
    
    with tab7:
        st.subheader(f"Top {TOP_N_ITEMS} Police Precincts by Incident Count")
        if not df_filtered.empty:
            precinct_fig = create_police_precinct_chart(df_filtered)
            st.plotly_chart(precinct_fig, use_container_width=True)
        else:
            st.warning("No data available for selected filters")
    
    with tab8:
        st.subheader(f"Top {TOP_N_ITEMS} Council Districts by Incident Count")
        if not df_filtered.empty:
            district_fig = create_council_district_chart(df_filtered)
            st.plotly_chart(district_fig, use_container_width=True)
        else:
            st.warning("No data available for selected filters")
    
    with tab9:
        st.subheader(f"Top {TOP_N_ITEMS} Neighborhoods by Incident Count")
        if not df_filtered.empty:
            neighborhood_fig = create_neighborhood_chart(df_filtered)
            st.plotly_chart(neighborhood_fig, use_container_width=True)
        else:
            st.warning("No data available for selected filters")
    
    with tab10:
        st.subheader(f"Top {TOP_N_ITEMS} Locations by Incident Count")
        if not df_filtered.empty:
            location_fig = create_location_chart(df_filtered)
            st.plotly_chart(location_fig, use_container_width=True)
        else:
            st.warning("No data available for selected filters")
    
    with tab11:
        st.subheader("1-Year Incident Forecast")
        if not df_filtered.empty:
            forecast_fig = create_forecast(df_filtered)
            if forecast_fig:
                st.plotly_chart(forecast_fig, use_container_width=True)
        else:
            st.warning("No data available for forecasting")
    
    with tab12:
        st.subheader("Pivot Analysis")
        if not df_filtered.empty:
            # Create columns for pivot controls
            col1, col2, col3 = st.columns([2, 2, 1])
            
            # Available columns for pivot analysis
            pivot_columns = PIVOT_COLUMNS
            
            # Initialize session state for pivot selections if not exists
            if "pivot_rows_selection" not in st.session_state:
                st.session_state.pivot_rows_selection = pivot_columns[0]
            if "pivot_columns_selection" not in st.session_state:
                st.session_state.pivot_columns_selection = pivot_columns[1]
            
            with col1:
                rows = st.selectbox("Rows (Group by):", pivot_columns, 
                                  index=pivot_columns.index(st.session_state.pivot_rows_selection) if st.session_state.pivot_rows_selection in pivot_columns else 0,
                                  key="pivot_rows")
                st.session_state.pivot_rows_selection = rows
            with col2:
                columns = st.selectbox("Columns:", pivot_columns, 
                                     index=pivot_columns.index(st.session_state.pivot_columns_selection) if st.session_state.pivot_columns_selection in pivot_columns else 1,
                                     key="pivot_columns")
                st.session_state.pivot_columns_selection = columns
            with col3:
                st.write("")  # Add some spacing
                if st.button("🔄 Swap", help="Swap rows and columns selections", key="swap_pivot"):
                    # Swap the selections without rerun
                    temp = st.session_state.pivot_rows_selection
                    st.session_state.pivot_rows_selection = st.session_state.pivot_columns_selection
                    st.session_state.pivot_columns_selection = temp
            
            # Create pivot table and chart (always count)
            values_col = 'incident_entry_id'  # Use incident ID for counting
            aggfunc = 'count'  # Always use count
            pivot_table, pivot_fig = create_pivot_chart(df_filtered, rows, columns, values_col, aggfunc)
            
            if pivot_table is not None and pivot_fig is not None:
                # Display pivot chart
                st.plotly_chart(pivot_fig, use_container_width=True)
                
                # Display pivot table
                st.subheader("Pivot Table Data")
                st.info(f"Table size: {pivot_table.shape[0]} rows × {pivot_table.shape[1]} columns")
                st.dataframe(pivot_table, use_container_width=True)
            else:
                st.error("Unable to create pivot table with selected parameters. Try different column combinations.")
        else:
            st.warning("No data available for selected filters")
    
    # Footer
    st.markdown("---")
    st.markdown("Data source: Detroit Open Data Portal | Last updated: " + 
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

if __name__ == "__main__":
    main()