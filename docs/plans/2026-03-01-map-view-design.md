# Map View Design

## Overview

Add a new "Map" page to the Detroit Crime dashboard showing incidents geographically with two modes: scatter (color-coded by offense category) and heatmap (density).

## Design

**New file**: `pages/5_Map.py`

**Two modes** via `st.radio` (horizontal):

1. **Scatter** — Each incident as a dot, color-coded by offense category using neon-bright colors. Hover shows category + nearest intersection. Sampled at `MAX_MAP_SAMPLE` (1000).
2. **Heatmap** — `density_mapbox` showing incident concentration. Neon color scale on dark basemap. Can use more points since density is aggregated.

**Common settings**:
- Basemap: `carto-darkmatter` (dark tiles, no API key)
- Center: Detroit (~42.3314, -83.0458), default zoom ~10
- Dark theme styling matching rest of app
- Respects sidebar offense category filter
- Zoom in/out via scroll/pinch

**Navigation**: Added as 5th page in `app.py` `st.navigation` list.

**No new dependencies** — uses Plotly scatter_mapbox and density_mapbox.
