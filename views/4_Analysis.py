"""Analysis page — interactive pivot table."""

import streamlit as st
import pandas as pd
import plotly.express as px

from constants import CHART_HEIGHT, PIVOT_COLUMNS, COLOR_SCALE_HEATMAP, CODE_LABEL_PREFIXES
from utils import code_labels, day_labels, numeric_codes
from theme import inject_css, apply_dark_plotly, TEAL_ACCENT

# Numeric fields shown on a continuous axis that should only tick on whole numbers
WHOLE_NUMBER_FIELDS = {"incident_year"}


def axis_labels(field: str, values) -> list[str] | None:
    """Names for a pivot axis, or None when its values are shown as they are.

    Precinct, district and zip codes get their prefix, and the feed's day
    numbers their names (a bare 1 could be Sunday or Monday).
    """
    if field in CODE_LABEL_PREFIXES:
        return code_labels(values, CODE_LABEL_PREFIXES[field])
    if field == "incident_day_of_week":
        return day_labels(values)
    return None

inject_css()

st.header("Pivot Analysis")

# The data app.py lends each run (see app.main)
run_data = st.session_state.get("run_data", {})
if "df_filtered" not in run_data:
    st.warning("Loading data — please wait…")
    st.stop()

df_filtered: pd.DataFrame = run_data["df_filtered"]

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

    # Counts rows. An incident with no value for one of the fields has no cell
    # to be counted in; how many there are is reported under the table.
    pivot_table = pd.pivot_table(df_sub, index=rows, columns=columns, aggfunc="size", fill_value=0)
    left_out = len(df_filtered) - int(pivot_table.to_numpy().sum())

    row_labels = axis_labels(rows, pivot_table.index)
    if row_labels is not None:
        pivot_table.index = pd.Index(row_labels, name=rows)
    column_labels = axis_labels(columns, pivot_table.columns)
    if column_labels is not None:
        pivot_table.columns = pd.Index(column_labels, name=columns)

    fig = px.imshow(
        pivot_table,
        labels=dict(x=columns, y=rows, color="Count"),
        aspect="auto",
        color_continuous_scale=COLOR_SCALE_HEATMAP,
    )
    # Labels are categories even when they look like numbers (zip codes).
    # Years are numbers: tick on whole values only, not 2024.5.
    if row_labels is not None:
        fig.update_yaxes(type="category")
    elif rows in WHOLE_NUMBER_FIELDS:
        fig.update_yaxes(dtick=1)
    if column_labels is not None:
        fig.update_xaxes(type="category")
    elif columns in WHOLE_NUMBER_FIELDS:
        fig.update_xaxes(dtick=1)
    fig.update_layout(height=CHART_HEIGHT)
    apply_dark_plotly(fig, TEAL_ACCENT)
    st.plotly_chart(fig, width="stretch")

    st.subheader("Pivot Table Data")
    st.info(f"Table size: {pivot_table.shape[0]} rows × {pivot_table.shape[1]} columns")
    if left_out:
        fields = " or ".join(dict.fromkeys([rows, columns]))
        st.caption(
            f"{left_out:,} of {len(df_filtered):,} incidents have no {fields} and are not in the table."
        )
    st.dataframe(pivot_table, width="stretch")

except Exception as e:
    st.error(f"Error creating pivot table: {str(e)}")
