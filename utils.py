"""Utility functions for Detroit Crime Dashboard

This module provides common utility functions for data loading, validation,
and processing used throughout the crime dashboard application.
"""

from typing import Optional, Tuple, List
import pandas as pd
import streamlit as st
import requests
from io import StringIO
from constants import REQUIRED_COLUMNS, CATEGORICAL_VARS


def validate_data(df: Optional[pd.DataFrame]) -> bool:
    """Validate that the dataframe has required columns and basic data quality"""
    if df is None or df.empty:
        raise ValueError("DataFrame is empty or None")
    
    # Check for required columns
    missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Check for basic data quality
    if df['latitude'].isna().all() or df['longitude'].isna().all():
        raise ValueError("All latitude/longitude values are missing")
    
    if df['incident_occurred_at'].isna().all():
        raise ValueError("All incident dates are missing")
    
    return True


def format_categorical_column(df: pd.DataFrame, column_name: str, prefix: str = "") -> pd.DataFrame:
    """
    Standardize categorical column formatting to remove decimals
    
    Args:
        df: DataFrame to process
        column_name: Name of column to format
        prefix: Optional prefix to add (e.g., "Precinct", "District")
    
    Returns:
        DataFrame with formatted column
    """
    df_copy = df.copy()
    
    if column_name not in df_copy.columns:
        return df_copy
    
    # Convert to numeric, then int, then string to remove decimals
    df_copy[column_name] = pd.to_numeric(df_copy[column_name], errors='coerce').fillna(0).astype(int).astype(str)
    
    # Remove invalid entries
    df_copy = df_copy[df_copy[column_name] != '0']
    
    # Add prefix if specified
    if prefix:
        df_copy[column_name] = df_copy[column_name].apply(lambda x: f"{prefix} {x}" if x != '0' else x)
    
    return df_copy


def format_zip_code_column(df: pd.DataFrame, column_name: str = 'zip_code') -> pd.DataFrame:
    """
    Format zip code column using string replacement method
    
    Args:
        df: DataFrame to process
        column_name: Name of zip code column
    
    Returns:
        DataFrame with formatted zip codes
    """
    df_copy = df.copy()
    
    if column_name not in df_copy.columns:
        return df_copy
    
    # More robust zip code formatting using string replacement
    df_copy[column_name] = df_copy[column_name].astype(str).str.replace('.0', '').str.replace('.', '')
    
    # Remove invalid entries
    invalid_values = ['0', 'nan', '', 'None']
    df_copy = df_copy[~df_copy[column_name].isin(invalid_values)]
    
    return df_copy


def create_generic_top_chart(df: pd.DataFrame, column_name: str, chart_title: str, 
                            y_label: str, top_n: int = 10, height: int = 400, 
                            categorical_format=None, prefix: str = ""):
    """
    Create a generic horizontal bar chart for top N analysis
    
    Args:
        df: DataFrame containing the data
        column_name: Column name to analyze
        chart_title: Title for the chart
        y_label: Label for y-axis
        top_n: Number of top items to show
        height: Chart height
        categorical_format: Special formatting function for categorical data
        prefix: Prefix to add to categorical labels
    
    Returns:
        Plotly figure object
    """
    import plotly.express as px
    
    # Apply categorical formatting if specified
    if categorical_format:
        df = categorical_format(df, column_name, prefix)
    
    # Get top N counts
    counts = df[column_name].value_counts().head(top_n)
    
    # Create horizontal bar chart
    fig = px.bar(
        x=counts.values[::-1],
        y=counts.index[::-1],
        orientation='h',
        labels={'x': 'Number of Incidents', 'y': y_label},
        title=chart_title
    )
    fig.update_layout(height=height)
    fig.update_yaxes(type='category')
    
    return fig


def safe_load_data_from_url(url: str, timeout: int = 30) -> pd.DataFrame:
    """
    Safely load data from URL with proper error handling
    
    Args:
        url: URL to fetch data from
        timeout: Request timeout in seconds
    
    Returns:
        DataFrame or None if failed
    
    Raises:
        Various specific exceptions for different failure modes
    """
    try:
        # Make request with timeout
        with st.spinner("Downloading Detroit crime data..."):
            response = requests.get(url, timeout=timeout)
            response.raise_for_status()
        
        # Parse CSV data
        df = pd.read_csv(StringIO(response.text), low_memory=False)
        
        return df
        
    except requests.exceptions.Timeout:
        raise requests.exceptions.Timeout("Request timed out. Please try again.")
    except requests.exceptions.ConnectionError:
        raise requests.exceptions.ConnectionError("Failed to connect to data source. Check your internet connection.")
    except requests.exceptions.HTTPError as e:
        raise requests.exceptions.HTTPError(f"HTTP error {e.response.status_code}: {e.response.reason}")
    except pd.errors.ParserError as e:
        raise pd.errors.ParserError(f"Failed to parse CSV data: {str(e)}")
    except Exception as e:
        raise Exception(f"Unexpected error loading data: {str(e)}")


def clean_and_filter_data(df: pd.DataFrame, min_date: str, 
                         lat_bounds: Tuple[float, float], 
                         lon_bounds: Tuple[float, float]) -> pd.DataFrame:
    """
    Clean and filter the crime data
    
    Args:
        df: Raw DataFrame
        min_date: Minimum date to include
        lat_bounds: Tuple of (min_lat, max_lat)
        lon_bounds: Tuple of (min_lon, max_lon)
    
    Returns:
        Cleaned DataFrame
    """
    # Clean and prepare data
    df = df.dropna(subset=['latitude', 'longitude'])
    
    # Convert date with error handling
    try:
        df['incident_occurred_at'] = pd.to_datetime(df['incident_occurred_at'])
    except (ValueError, TypeError) as e:
        raise ValueError(f"Failed to parse incident dates: {str(e)}")
    
    df['date'] = df['incident_occurred_at'].dt.date
    
    # Filter for data from min_date forward
    df = df[df['incident_occurred_at'] >= min_date]
    
    # Find the most recent date and exclude it (incomplete data)
    most_recent_date = df['date'].max()
    df = df[df['date'] < most_recent_date]
    
    # Filter out records with invalid coordinates
    df = df[(df['latitude'] != 0) & (df['longitude'] != 0)]
    df = df[(df['latitude'].between(lat_bounds[0], lat_bounds[1])) & 
            (df['longitude'].between(lon_bounds[0], lon_bounds[1]))]
    
    return df