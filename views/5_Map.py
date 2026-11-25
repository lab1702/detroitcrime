"""Map page — geographic incident visualization with scatter and heatmap modes."""

import html
import json

import streamlit as st
import pandas as pd
import pydeck as pdk
from pydeck.bindings.json_tools import default_serialize

from constants import MAX_SCATTER_POINTS
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

# Keep only what the map needs, and drop rows missing coordinates
df_map = df_filtered[
    ["latitude", "longitude", "offense_category", "nearest_intersection"]
].dropna(subset=["latitude", "longitude"])

if df_map.empty:
    st.warning("No geocoded incidents available.")
    st.stop()

# ── Controls ─────────────────────────────────────────────────────────────────
# Streamlit drops a widget's state while its page is not shown, so the chosen
# view is also kept under a plain session-state key and used to seed the
# widget when the user comes back to this page.
VIEWS = ["Scatter", "Heatmap"]
if "map_view" not in st.session_state:
    st.session_state["map_view"] = st.session_state.get("map_view_selection", VIEWS[0])
mode = st.radio("View", VIEWS, horizontal=True, key="map_view")
st.session_state["map_view_selection"] = mode

# Detroit center
DETROIT_LAT = 42.3314
DETROIT_LON = -83.0458

# Neon color palette for offense categories (RGB)
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
# Shared by every category beyond the palette, shown as "Other" in the legend
OTHER_COLOR = [140, 150, 165]  # grey

MAP_HEIGHT = 700


class CompactDeck(pdk.Deck):
    """A Deck that serializes without pretty-printing.

    pydeck indents its JSON, and with one object per point the whitespace alone
    is about half of what gets sent to the browser.
    """

    def to_json(self) -> str:
        return json.dumps(self, sort_keys=True, default=default_serialize, separators=(",", ":"))

# Assign colors from the full dataset's category ranking (most frequent first),
# so a category keeps its color whatever the filters are and no two categories
# named in the legend share one.
category_order = (
    st.session_state.get("category_order")
    or df_map["offense_category"].value_counts().index.tolist()
)
color_map = dict(zip(category_order, NEON_COLORS))

view_state = pdk.ViewState(
    latitude=DETROIT_LAT,
    longitude=DETROIT_LON,
    zoom=10,
    pitch=0,
)

total_points = len(df_map)
is_sampled = mode == "Scatter" and total_points > MAX_SCATTER_POINTS

if mode == "Scatter":
    # Every point is sent to the browser as JSON, so very wide selections are
    # sampled and the payload is kept small (short keys, rounded coordinates).
    df_points = df_map.sample(n=MAX_SCATTER_POINTS, random_state=0) if is_sampled else df_map
    category = df_points["offense_category"]

    def color_channel(index: int) -> pd.Series:
        channel = {name: rgb[index] for name, rgb in color_map.items()}
        return category.map(channel).fillna(OTHER_COLOR[index]).astype(int)

    # Plain columns only; missing text would serialize as NaN, which is not valid JSON
    df_scatter = pd.DataFrame({
        "lat": df_points["latitude"].round(5),
        "lon": df_points["longitude"].round(5),
        "category": category.fillna("Unknown"),
        "location": df_points["nearest_intersection"].fillna("Unknown location"),
        "r": color_channel(0),
        "g": color_channel(1),
        "b": color_channel(2),
    })

    layer = pdk.Layer(
        "ScatterplotLayer",
        data=df_scatter,
        get_position=["lon", "lat"],
        get_color="[r, g, b, 180]",
        get_radius=40,
        radius_min_pixels=2,
        radius_max_pixels=8,
        opacity=0.7,
        pickable=True,
    )
    # A text tooltip (not HTML), so values from the feed are never parsed as markup
    tooltip = {
        "text": "{category}\n{location}",
        "style": {
            "backgroundColor": "#0a0e14",
            "color": "#e0e0e0",
            "border": "1px solid #2d3548",
            "borderRadius": "4px",
            "padding": "8px",
        },
    }

else:
    # Incidents at the same spot are sent once with a weight, which gives the
    # same density as one row per incident with a far smaller payload.
    df_heat = (
        pd.DataFrame({
            "lat": df_map["latitude"].round(4),
            "lon": df_map["longitude"].round(4),
        })
        .value_counts()
        .reset_index(name="weight")
    )

    layer = pdk.Layer(
        "HeatmapLayer",
        data=df_heat,
        get_position=["lon", "lat"],
        get_weight="weight",
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

deck = CompactDeck(
    layers=[layer],
    initial_view_state=view_state,
    map_style="https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json",
    height=MAP_HEIGHT,
    tooltip=tooltip,
)

st.pydeck_chart(deck, width="stretch", height=MAP_HEIGHT)

# ── Legend for scatter mode ──────────────────────────────────────────────────
if mode == "Scatter":
    # Build a compact CSS legend
    visible_cats = sorted(df_map["offense_category"].dropna().unique())
    legend_entries = [(cat, color_map[cat]) for cat in visible_cats if cat in color_map]
    other_count = sum(1 for cat in visible_cats if cat not in color_map)
    if other_count:
        noun = "category" if other_count == 1 else "categories"
        legend_entries.append((f"Other ({other_count} {noun})", OTHER_COLOR))

    legend_items = []
    for label, (r, g, b) in legend_entries:
        legend_items.append(
            f'<span style="display:inline-block;width:10px;height:10px;'
            f'background:rgb({r},{g},{b});border-radius:50%;margin-right:5px;"></span>'
            f'<span style="margin-right:14px;">{html.escape(label)}</span>'
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
if is_sampled:
    st.caption(
        f"Showing a random sample of {MAX_SCATTER_POINTS:,} of {total_points:,} geocoded "
        "incidents. Narrow the filters to plot every point, or switch to Heatmap, which "
        "always uses all of them."
    )
else:
    st.caption(f"{total_points:,} geocoded incidents")
