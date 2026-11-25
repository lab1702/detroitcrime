"""Analysis page — interactive pivot table."""

import streamlit as st
import pandas as pd
import plotly.express as px

from constants import CHART_HEIGHT, PIVOT_COLUMNS, COLOR_SCALE_HEATMAP, CODE_LABEL_PREFIXES
from utils import code_labels, numeric_codes
from theme import inject_css, apply_dark_plotly, TEAL_ACCENT

# Numeric fields shown on a continuous axis that should only tick on whole numbers
WHOLE_NUMBER_FIELDS = {"incident_year", "incident_day_of_week"}

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
# Streamlit drops a widget's state while its page is not shown, so the current
# selection is also kept under plain session-state keys and used to seed the
# widgets when the user comes back to this page.
if "pivot_rows" not in st.session_state:
    st.session_state["pivot_rows"] = st.session_state.get("pivot_rows_selection", PIVOT_COLUMNS[0])
if "pivot_columns" not in st.session_state:
    st.session_state["pivot_columns"] = st.session_state.get("pivot_columns_selection", PIVOT_COLUMNS[1])


def swap_pivot_axes() -> None:
    """Swap the row and column selections (runs before the page is redrawn)."""
    st.session_state["pivot_rows"], st.session_state["pivot_columns"] = (
        st.session_state["pivot_columns"],
        st.session_state["pivot_rows"],
    )


col1, col2, col3 = st.columns([2, 2, 1])

with col1:
    rows = st.selectbox("Rows (Group by):", PIVOT_COLUMNS, key="pivot_rows")

with col2:
    columns = st.selectbox("Columns:", PIVOT_COLUMNS, key="pivot_columns")

with col3:
    st.write("")
    st.button("Swap", help="Swap rows and columns", key="swap_pivot", on_click=swap_pivot_axes)

st.session_state["pivot_rows_selection"] = rows
st.session_state["pivot_columns_selection"] = columns

# ── Create pivot table ───────────────────────────────────────────────────────
try:
    df_sub = df_filtered[list({rows, columns})]

    # Precinct, district and zip codes are pivoted as numbers, so they sort as
    # numbers (Precinct 2 before Precinct 10), and are labelled afterwards.
    # Rows without a valid code are left out.
    for field in {rows, columns} & CODE_LABEL_PREFIXES.keys():
        codes = numeric_codes(df_sub[field])
        df_sub = df_sub.assign(**{field: codes})[codes != 0]

    # Counts rows, so every incident in the selection is included
    pivot_table = pd.pivot_table(df_sub, index=rows, columns=columns, aggfunc="size", fill_value=0)

    if rows in CODE_LABEL_PREFIXES:
        labels = code_labels(pivot_table.index, CODE_LABEL_PREFIXES[rows])
        pivot_table.index = pd.Index(labels, name=rows)
    if columns in CODE_LABEL_PREFIXES:
        labels = code_labels(pivot_table.columns, CODE_LABEL_PREFIXES[columns])
        pivot_table.columns = pd.Index(labels, name=columns)

    fig = px.imshow(
        pivot_table,
        labels=dict(x=columns, y=rows, color="Count"),
        aspect="auto",
        color_continuous_scale=COLOR_SCALE_HEATMAP,
    )
    # Code labels are categories even when they look like numbers (zip codes).
    # Years and weekdays are numbers: tick on whole values only, not 2024.5.
    if rows in CODE_LABEL_PREFIXES:
        fig.update_yaxes(type="category")
    elif rows in WHOLE_NUMBER_FIELDS:
        fig.update_yaxes(dtick=1)
    if columns in CODE_LABEL_PREFIXES:
        fig.update_xaxes(type="category")
    elif columns in WHOLE_NUMBER_FIELDS:
        fig.update_xaxes(dtick=1)
    fig.update_layout(height=CHART_HEIGHT)
    apply_dark_plotly(fig, TEAL_ACCENT)
    st.plotly_chart(fig, width="stretch")

    st.subheader("Pivot Table Data")
    st.info(f"Table size: {pivot_table.shape[0]} rows × {pivot_table.shape[1]} columns")
    st.dataframe(pivot_table, width="stretch")

except Exception as e:
    st.error(f"Error creating pivot table: {str(e)}")
