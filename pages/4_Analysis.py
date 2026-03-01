"""Analysis page — interactive pivot table."""

import streamlit as st
import pandas as pd
import plotly.express as px

from constants import CHART_HEIGHT, PIVOT_COLUMNS, COLOR_SCALE_HEATMAP, CATEGORICAL_VARS
from utils import format_categorical_column, format_zip_code_column
from theme import inject_css, apply_dark_plotly, TEAL_ACCENT

inject_css()

st.header("Pivot Analysis")

if "df_filtered" not in st.session_state:
    st.warning("Loading data — please wait…")
    st.stop()

df_filtered: pd.DataFrame = st.session_state["df_filtered"]

if df_filtered.empty:
    st.warning("No data available for selected filters.")
    st.stop()

# ── Pivot controls ───────────────────────────────────────────────────────────
col1, col2, col3 = st.columns([2, 2, 1])

if "pivot_rows_selection" not in st.session_state:
    st.session_state.pivot_rows_selection = PIVOT_COLUMNS[0]
if "pivot_columns_selection" not in st.session_state:
    st.session_state.pivot_columns_selection = PIVOT_COLUMNS[1]

with col1:
    rows = st.selectbox(
        "Rows (Group by):",
        PIVOT_COLUMNS,
        index=(
            PIVOT_COLUMNS.index(st.session_state.pivot_rows_selection)
            if st.session_state.pivot_rows_selection in PIVOT_COLUMNS
            else 0
        ),
        key="pivot_rows",
    )
    st.session_state.pivot_rows_selection = rows

with col2:
    columns = st.selectbox(
        "Columns:",
        PIVOT_COLUMNS,
        index=(
            PIVOT_COLUMNS.index(st.session_state.pivot_columns_selection)
            if st.session_state.pivot_columns_selection in PIVOT_COLUMNS
            else 1
        ),
        key="pivot_columns",
    )
    st.session_state.pivot_columns_selection = columns

with col3:
    st.write("")
    if st.button("Swap", help="Swap rows and columns", key="swap_pivot"):
        temp = st.session_state.pivot_rows_selection
        st.session_state.pivot_rows_selection = st.session_state.pivot_columns_selection
        st.session_state.pivot_columns_selection = temp

# ── Create pivot table ───────────────────────────────────────────────────────
try:
    required_cols = list(set([rows, columns, "incident_entry_id"]))
    df_sub = df_filtered[required_cols].copy()

    if rows in CATEGORICAL_VARS:
        prefix = "Precinct" if rows == "police_precinct" else "District" if rows == "council_district" else ""
        if rows == "zip_code":
            df_sub = format_zip_code_column(df_sub, rows)
        else:
            df_sub = format_categorical_column(df_sub, rows, prefix)

    if columns in CATEGORICAL_VARS and columns != rows:
        prefix = "Precinct" if columns == "police_precinct" else "District" if columns == "council_district" else ""
        if columns == "zip_code":
            df_sub = format_zip_code_column(df_sub, columns)
        else:
            df_sub = format_categorical_column(df_sub, columns, prefix)

    pivot_table = pd.pivot_table(
        df_sub, index=rows, columns=columns, values="incident_entry_id",
        aggfunc="count", fill_value=0,
    )

    if rows == "zip_code":
        pivot_table.index = pivot_table.index.astype(str)
    if columns == "zip_code":
        pivot_table.columns = pivot_table.columns.astype(str)

    fig = px.imshow(
        pivot_table,
        labels=dict(x=columns, y=rows, color="Count"),
        aspect="auto",
        color_continuous_scale=COLOR_SCALE_HEATMAP,
    )
    if rows == "zip_code":
        fig.update_yaxes(type="category")
    if columns == "zip_code":
        fig.update_xaxes(type="category")
    fig.update_layout(height=CHART_HEIGHT)
    apply_dark_plotly(fig, TEAL_ACCENT)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Pivot Table Data")
    st.info(f"Table size: {pivot_table.shape[0]} rows × {pivot_table.shape[1]} columns")
    st.dataframe(pivot_table, use_container_width=True)

except Exception as e:
    st.error(f"Error creating pivot table: {str(e)}")
