"""Map page — geographic incident visualization with scatter and heatmap modes."""

import streamlit as st
import pandas as pd
import pydeck as pdk

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

# Neon color palette for offense categories (RGBA)
NEON_COLORS = [
    [255, 75, 75],    # red
    [0, 212, 255],    # cyan
    [255, 170, 0],    # amber
    [0, 200, 83],     # green
    [224, 64, 251],   # magenta
    [255, 110, 64],   # deep orange
    [0, 191, 165],    # teal
    [255, 234, 0],    # yellow
    [124, 77, 255],   # purple
    [24, 255, 255],   # light cyan
    [255, 128, 171],  # pink
    [118, 255, 3],    # lime
    [245, 0, 87],     # hot pink
    [101, 31, 255],   # deep purple
    [0, 229, 255],    # bright cyan
]

MAP_HEIGHT = 700

# Assign colors to categories based on the full dataset (stable mapping)
all_categories = sorted(df_map["offense_category"].unique())
color_map = {cat: NEON_COLORS[i % len(NEON_COLORS)] for i, cat in enumerate(all_categories)}
df_map = df_map.copy()
df_map["color_r"] = df_map["offense_category"].map(lambda c: color_map[c][0])
df_map["color_g"] = df_map["offense_category"].map(lambda c: color_map[c][1])
df_map["color_b"] = df_map["offense_category"].map(lambda c: color_map[c][2])

view_state = pdk.ViewState(
    latitude=DETROIT_LAT,
    longitude=DETROIT_LON,
    zoom=10,
    pitch=0,
)

if mode == "Scatter":
    layer = pdk.Layer(
        "ScatterplotLayer",
        data=df_map,
        get_position=["longitude", "latitude"],
        get_color="[color_r, color_g, color_b, 180]",
        get_radius=40,
        radius_min_pixels=2,
        radius_max_pixels=8,
        opacity=0.7,
        pickable=True,
    )
    tooltip = {
        "html": "<b>{offense_category}</b><br/>{nearest_intersection}",
        "style": {
            "backgroundColor": "#0a0e14",
            "color": "#e0e0e0",
            "border": "1px solid #2d3548",
            "borderRadius": "4px",
            "padding": "8px",
        },
    }

else:
    layer = pdk.Layer(
        "HeatmapLayer",
        data=df_map,
        get_position=["longitude", "latitude"],
        get_weight=1,
        radiusPixels=30,
        intensity=1,
        threshold=0.1,
        color_range=[
            [13, 0, 48],      # deep dark purple
            [106, 13, 173],   # purple
            [224, 64, 251],   # magenta
            [255, 110, 64],   # deep orange
            [255, 234, 0],    # yellow
        ],
        opacity=0.8,
    )
    tooltip = None

deck = pdk.Deck(
    layers=[layer],
    initial_view_state=view_state,
    map_style="https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json",
    height=MAP_HEIGHT,
)

st.pydeck_chart(deck, use_container_width=True, height=MAP_HEIGHT)

# ── Legend for scatter mode ──────────────────────────────────────────────────
if mode == "Scatter":
    # Build a compact CSS legend
    visible_cats = sorted(df_map["offense_category"].unique())
    legend_items = []
    for cat in visible_cats:
        r, g, b = color_map[cat]
        legend_items.append(
            f'<span style="display:inline-block;width:10px;height:10px;'
            f'background:rgb({r},{g},{b});border-radius:50%;margin-right:5px;"></span>'
            f'<span style="margin-right:14px;">{cat}</span>'
        )
    legend_html = (
        '<div style="background:#0a0e14;border:1px solid #2d3548;border-radius:6px;'
        'padding:10px 14px;color:#c0c8d4;font-size:0.8rem;line-height:1.8;'
        'max-height:150px;overflow-y:auto;">'
        + "".join(legend_items)
        + "</div>"
    )
    st.markdown(legend_html, unsafe_allow_html=True)

# ── Info line ────────────────────────────────────────────────────────────────
st.caption(f"{len(df_map):,} geocoded incidents")
