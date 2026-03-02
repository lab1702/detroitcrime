"""Map page — geographic incident visualization with scatter and heatmap modes."""

import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import plotly.express as px

from constants import MAX_MAP_SAMPLE
from theme import inject_css

inject_css()

st.header("Map")

if "df_filtered" not in st.session_state:
    st.warning("Loading data — please wait…")
    st.stop()

df_filtered: pd.DataFrame = st.session_state["df_filtered"]

if df_filtered.empty:
    st.warning("No data available for selected filters.")
    st.stop()

# Drop rows missing coordinates
df_map = df_filtered.dropna(subset=["latitude", "longitude"])

if df_map.empty:
    st.warning("No geocoded incidents available.")
    st.stop()

# ── Controls ─────────────────────────────────────────────────────────────────
mode = st.radio("View", ["Scatter", "Heatmap"], horizontal=True)

# Detroit center
DETROIT_LAT = 42.3314
DETROIT_LON = -83.0458

# Neon color palette for offense categories
NEON_COLORS = [
    "#ff4b4b",  # red
    "#00d4ff",  # cyan
    "#ffaa00",  # amber
    "#00c853",  # green
    "#e040fb",  # magenta
    "#ff6e40",  # deep orange
    "#00bfa5",  # teal
    "#ffea00",  # yellow
    "#7c4dff",  # purple
    "#18ffff",  # light cyan
    "#ff80ab",  # pink
    "#76ff03",  # lime
    "#f50057",  # hot pink
    "#651fff",  # deep purple
    "#00e5ff",  # bright cyan
]

MAP_HEIGHT = 700

# JavaScript to prevent page scroll when wheeling over the map.
# The Plotly chart lives in an iframe; wheel events on the iframe element
# in the parent document cause the Streamlit container to scroll.
# We intercept those events so scroll-wheel zooms the map instead.
_SCROLL_FIX_JS = """
<script>
(function() {
    const doc = window.parent.document;
    function fix() {
        doc.querySelectorAll('[data-testid="stPlotlyChart"]').forEach(function(el) {
            if (!el._mapScrollFixed) {
                el._mapScrollFixed = true;
                el.addEventListener('wheel', function(e) {
                    e.preventDefault();
                }, {passive: false});
            }
        });
    }
    fix();
    setTimeout(fix, 500);
    setTimeout(fix, 1500);
})();
</script>
"""

if mode == "Scatter":
    # Sample to keep rendering fast
    if len(df_map) > MAX_MAP_SAMPLE:
        df_plot = df_map.sample(n=MAX_MAP_SAMPLE, random_state=42)
    else:
        df_plot = df_map

    # Assign neon colors to categories
    categories = sorted(df_plot["offense_category"].unique())
    color_map = {cat: NEON_COLORS[i % len(NEON_COLORS)] for i, cat in enumerate(categories)}

    fig = px.scatter_mapbox(
        df_plot,
        lat="latitude",
        lon="longitude",
        color="offense_category",
        color_discrete_map=color_map,
        hover_data={"nearest_intersection": True, "offense_category": True,
                     "latitude": False, "longitude": False},
        opacity=0.7,
        zoom=10,
        center={"lat": DETROIT_LAT, "lon": DETROIT_LON},
    )
    fig.update_layout(
        mapbox_style="carto-darkmatter",
        height=MAP_HEIGHT,
        paper_bgcolor="#0E1117",
        font=dict(color="#c0c8d4"),
        margin=dict(l=0, r=0, t=0, b=0),
        legend=dict(
            title="Offense Category",
            bgcolor="rgba(10, 14, 20, 0.85)",
            bordercolor="#2d3548",
            borderwidth=1,
            font=dict(color="#c0c8d4", size=11),
        ),
    )
    fig.update_traces(marker=dict(size=5))
    st.plotly_chart(fig, use_container_width=True, config={"scrollZoom": True})
    components.html(_SCROLL_FIX_JS, height=0)

else:
    # Heatmap can handle more points since it's aggregated
    max_heatmap = MAX_MAP_SAMPLE * 5
    if len(df_map) > max_heatmap:
        df_plot = df_map.sample(n=max_heatmap, random_state=42)
    else:
        df_plot = df_map

    fig = px.density_mapbox(
        df_plot,
        lat="latitude",
        lon="longitude",
        radius=8,
        zoom=10,
        center={"lat": DETROIT_LAT, "lon": DETROIT_LON},
        color_continuous_scale=[
            [0.0, "#0d0030"],
            [0.25, "#6a0dad"],
            [0.5, "#e040fb"],
            [0.75, "#ff6e40"],
            [1.0, "#ffea00"],
        ],
    )
    fig.update_layout(
        mapbox_style="carto-darkmatter",
        height=MAP_HEIGHT,
        paper_bgcolor="#0E1117",
        font=dict(color="#c0c8d4"),
        margin=dict(l=0, r=0, t=0, b=0),
        coloraxis_colorbar=dict(
            title=dict(text="Density", font=dict(color="#c0c8d4")),
            tickfont=dict(color="#c0c8d4"),
            bgcolor="rgba(10, 14, 20, 0.85)",
            bordercolor="#2d3548",
            borderwidth=1,
        ),
    )
    st.plotly_chart(fig, use_container_width=True, config={"scrollZoom": True})
    components.html(_SCROLL_FIX_JS, height=0)

# ── Info line ────────────────────────────────────────────────────────────────
sample_note = ""
if mode == "Scatter" and len(df_map) > MAX_MAP_SAMPLE:
    sample_note = f" (showing {MAX_MAP_SAMPLE:,} of {len(df_map):,})"
elif mode == "Heatmap" and len(df_map) > MAX_MAP_SAMPLE * 5:
    sample_note = f" (showing {MAX_MAP_SAMPLE * 5:,} of {len(df_map):,})"

st.caption(f"{len(df_map):,} geocoded incidents{sample_note}")
