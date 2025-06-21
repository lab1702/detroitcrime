import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import folium
from streamlit_folium import folium_static
from prophet import Prophet
from datetime import datetime

# Configure page
st.set_page_config(
    page_title="Detroit Crime Incidents",
    page_icon="🚔",
    layout="wide"
)

# Cache for 24 hours (86400 seconds)
@st.cache_data(ttl=86400)
def load_data():
    """Load Detroit crime data with 24-hour caching"""
    url = "https://data.detroitmi.gov/api/download/v1/items/8e532daeec1149879bd5e67fdd9c8be0/csv?layers=0"
    
    with st.spinner("Loading Detroit crime data..."):
        try:
            df = pd.read_csv(url)
            
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
                  title='Daily Crime Incident Count',
                  labels={'date': 'Date', 'count': 'Number of Incidents'})
    fig.update_layout(height=800)
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
                    title='Crime Incidents Heatmap: Day of Week vs Hour of Day',
                    labels=dict(x="Hour of Day", y="Day of Week", color="Incident Count"),
                    aspect="auto",
                    color_continuous_scale="Reds")
    
    fig.update_yaxes(tickvals=list(range(1, 8)), ticktext=day_labels)
    fig.update_layout(height=800)
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
        
        # Create and fit model
        model = Prophet(daily_seasonality=False, weekly_seasonality=True, yearly_seasonality=True)
        model.fit(daily_counts)
        
        # Make future dataframe for 1 year (365 days)
        future = model.make_future_dataframe(periods=365)
        forecast = model.predict(future)
        
        # Create forecast plot
        fig = go.Figure()
        
        # Historical data
        fig.add_trace(go.Scatter(
            x=daily_counts['ds'], 
            y=daily_counts['y'],
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
        
        fig.update_layout(
            title='1-Year Crime Incident Forecast',
            xaxis_title='Date',
            yaxis_title='Incident Count',
            height=400
        )
        
        return fig
    except Exception as e:
        st.error(f"Error creating forecast: {str(e)}")
        return None

def create_offense_category_chart(df):
    """Create chart showing distribution of offense categories"""
    offense_counts = df['offense_category'].value_counts().head(10)
    
    fig = px.bar(
        x=offense_counts.values[::-1],
        y=offense_counts.index[::-1],
        orientation='h',
        title='Top 10 Crime Categories',
        labels={'x': 'Number of Incidents', 'y': 'Offense Category'}
    )
    fig.update_layout(height=800)
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
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Map", "Time Series", "Heatmaps", "Forecast", "Categories"])
    
    with tab1:
        st.subheader("Crime Incident Map")
        if not df_filtered.empty:
            map_obj = create_map(df_filtered)
            folium_static(map_obj, width=1000, height=600)
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
        st.subheader("Day of Week vs Hour of Day Heatmap")
        if not df_filtered.empty:
            heatmap_fig = create_day_of_week_heatmap(df_filtered)
            st.plotly_chart(heatmap_fig, use_container_width=True)
        else:
            st.warning("No data available for selected filters")
    
    with tab4:
        st.subheader("1-Year Incident Forecast")
        if not df_filtered.empty:
            forecast_fig = create_forecast(df_filtered)
            if forecast_fig:
                st.plotly_chart(forecast_fig, use_container_width=True)
        else:
            st.warning("No data available for forecasting")
    
    with tab5:
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