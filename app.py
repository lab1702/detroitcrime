import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import folium
from streamlit_folium import st_folium
from prophet import Prophet
from prophet.make_holidays import make_holidays_df
from datetime import datetime

# Configure page
st.set_page_config(
    page_title="Detroit Crime Incidents",
    page_icon="🚔",
    layout="wide"
)

# Cache for 24 hours (86400 seconds)
@st.cache_data(ttl=86400, show_spinner=False)
def load_data():
    """Load Detroit crime data with 24-hour caching"""
    url = "https://data.detroitmi.gov/api/download/v1/items/8e532daeec1149879bd5e67fdd9c8be0/csv?layers=0"
    
    with st.spinner("Downloading Detroit crime data..."):
        try:
            df = pd.read_csv(url, low_memory=False)
            
            # Clean and prepare data
            df = df.dropna(subset=['latitude', 'longitude'])
            df['incident_occurred_at'] = pd.to_datetime(df['incident_occurred_at'])
            df['date'] = df['incident_occurred_at'].dt.date
            
            # Filter for data from 2018-01-01 forward
            df = df[df['incident_occurred_at'] >= '2018-01-01']
            
            # Find the most recent date and exclude it
            most_recent_date = df['date'].max()
            df = df[df['date'] < most_recent_date]
            
            # Filter out records with invalid coordinates
            df = df[(df['latitude'] != 0) & (df['longitude'] != 0)]
            df = df[(df['latitude'].between(42.0, 43.0)) & (df['longitude'].between(-84.0, -82.0))]
            
            return df
        except Exception as e:
            st.error(f"Error loading data: {str(e)}")
            return None

def create_map(df, sample_size=1000):
    """Create interactive map of crime incidents"""
    # Sample data for performance
    if len(df) > sample_size:
        df_sample = df.sample(n=sample_size, random_state=42)
    else:
        df_sample = df
    
    # Create map centered on Detroit
    detroit_lat, detroit_lon = 42.3314, -83.0458
    m = folium.Map(location=[detroit_lat, detroit_lon], zoom_start=11)
    
    # Add markers
    for _, row in df_sample.iterrows():
        folium.CircleMarker(
            location=[row['latitude'], row['longitude']],
            radius=3,
            popup=f"{row['offense_category']}<br>{row['nearest_intersection']}<br>{row['incident_occurred_at'].strftime('%Y-%m-%d %H:%M')}",
            color='red',
            fillColor='red',
            fillOpacity=0.6
        ).add_to(m)
    
    return m

def create_time_series_chart(df):
    """Create historical incident count by day"""
    daily_counts = df.groupby('date').size().reset_index(name='count')
    daily_counts['date'] = pd.to_datetime(daily_counts['date'])
    
    fig = px.line(daily_counts, x='date', y='count', 
                  labels={'date': 'Date', 'count': 'Number of Incidents'})
    fig.update_layout(height=600)
    return fig

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
                    color_continuous_scale="Reds")
    
    fig.update_yaxes(tickvals=list(range(1, 8)), ticktext=day_labels)
    fig.update_layout(height=600)
    return fig

def create_forecast(df):
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
        
        # Historical data - only show last 1 year before forecast
        forecast_start = daily_counts['ds'].max()
        one_year_before = forecast_start - pd.DateOffset(years=1)
        historical_data = daily_counts[daily_counts['ds'] >= one_year_before]
        
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
            height=600
        )
        
        return fig
    except Exception as e:
        st.error(f"Error creating forecast: {str(e)}")
        return None

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
        
        # Check if we have at least 2 months of data (approximately 60 days)
        if days_elapsed >= 60:
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
    
    fig.update_layout(height=600, barmode='overlay')
    return fig

def create_offense_category_chart(df):
    """Create chart showing distribution of offense categories"""
    offense_counts = df['offense_category'].value_counts().head(10)
    
    fig = px.bar(
        x=offense_counts.values[::-1],
        y=offense_counts.index[::-1],
        orientation='h',
        labels={'x': 'Number of Incidents', 'y': 'Offense Category'}
    )
    fig.update_layout(height=600)
    return fig

def main():
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
    
    # Filter data
    df_filtered = df
    
    if selected_category != 'All':
        df_filtered = df_filtered[df_filtered['offense_category'] == selected_category]
    
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
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["Map", "Time Series", "Year over Year", "Heatmaps", "Forecast", "Categories"])
    
    with tab1:
        st.subheader("Crime Incident Map")
        if not df_filtered.empty:
            map_obj = create_map(df_filtered)
            st_folium(map_obj, width=1000, height=600)
        else:
            st.warning("No data available for selected filters")
    
    with tab2:
        st.subheader("Historical Incident Count")
        if not df_filtered.empty:
            time_series_fig = create_time_series_chart(df_filtered)
            st.plotly_chart(time_series_fig, use_container_width=True)
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
        st.subheader("Day of Week vs Hour of Day Heatmap")
        if not df_filtered.empty:
            heatmap_fig = create_day_of_week_heatmap(df_filtered)
            st.plotly_chart(heatmap_fig, use_container_width=True)
        else:
            st.warning("No data available for selected filters")
    
    with tab5:
        st.subheader("1-Year Incident Forecast")
        if not df_filtered.empty:
            forecast_fig = create_forecast(df_filtered)
            if forecast_fig:
                st.plotly_chart(forecast_fig, use_container_width=True)
        else:
            st.warning("No data available for forecasting")
    
    with tab6:
        st.subheader("Crime Categories")
        if selected_category != 'All':
            st.info("Select 'All' in the offense category filter to view the top 10 crime categories chart.")
        elif not df_filtered.empty:
            category_fig = create_offense_category_chart(df_filtered)
            st.plotly_chart(category_fig, use_container_width=True)
        else:
            st.warning("No data available for selected filters")
    
    # Footer
    st.markdown("---")
    st.markdown("Data source: Detroit Open Data Portal | Last updated: " + 
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

if __name__ == "__main__":
    main()