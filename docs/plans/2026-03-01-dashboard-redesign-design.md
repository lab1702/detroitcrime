# Detroit Crime Dashboard Redesign

## Goals

- Visual polish: dark, data-dense "command center" aesthetic
- Better UX: overview-first information hierarchy with multi-page navigation
- Contextual color: red/amber for severity, blue/cyan for trends, green/teal for positive metrics

## App Structure

Multi-page Streamlit app using `pages/` directory convention:

```
app.py                  # Entry point, shared data loading & theming
pages/
  1_Overview.py         # Landing page: KPIs + summary charts
  2_Trends.py           # Daily incidents, YoY, heatmap, forecast
  3_Geography.py        # Precincts, districts, neighborhoods, zips, locations
  4_Analysis.py         # Pivot table
```

Shared data loading (cached) in `app.py`, accessed via `st.session_state`. Offense category filter in sidebar, applied globally. `utils.py` and `constants.py` unchanged.

## Overview Page (Landing)

```
┌─────────────────────────────────────────────────────┐
│  DETROIT CRIME INCIDENTS                            │
├────────────┬────────────┬────────────┬──────────────┤
│ TOTAL      │ DATE RANGE │ DAILY AVG  │ TOP CATEGORY │
│ 142,387    │ 2,190 days │ 65.0/day   │ Assault      │
│ +3.2%      │            │ -1.4%      │              │
├────────────┴────────────┴────────────┴──────────────┤
│  ┌──────────────────────┐ ┌───────────────────────┐ │
│  │  TOP 10 CATEGORIES   │ │   CASE STATUS         │ │
│  │  (horizontal bar)    │ │   (horizontal bar)    │ │
│  └──────────────────────┘ └───────────────────────┘ │
│  ┌──────────────────────────────────────────────────┤
│  │  DAILY INCIDENTS (30-day trend sparkline)       │ │
│  └──────────────────────────────────────────────────┤
│  footer: Data source + last updated                 │
└─────────────────────────────────────────────────────┘
```

- 4 KPI cards: large numbers, dark card styling with subtle border glow
- YoY % change indicators where applicable
- Categories + case status side-by-side (2 columns)
- Compact daily trend sparkline below

## Trends Page

```
┌──────────────────────────────────────────────────────┐
│  ┌──────────────────────────────────────────────────┐│
│  │  DAILY INCIDENTS (full time series, full width)  ││
│  └──────────────────────────────────────────────────┘│
│  ┌─────────────────────┐ ┌──────────────────────────┐│
│  │  YEAR OVER YEAR     │ │  HOUR x DAY HEATMAP     ││
│  │  (bar + projections)│ │  (7x24 grid)            ││
│  └─────────────────────┘ └──────────────────────────┘│
│  ┌──────────────────────────────────────────────────┐│
│  │  FORECAST (Prophet, 1-year, full width)          ││
│  │  confidence intervals + holiday annotations      ││
│  └──────────────────────────────────────────────────┘│
└──────────────────────────────────────────────────────┘
```

- Blue/cyan accent colors
- Daily incidents full width (primary view)
- YoY + heatmap side-by-side
- Forecast full width at bottom

## Geography Page

```
┌──────────────────────────────────────────────────────┐
│  ┌─────────────────────┐ ┌──────────────────────────┐│
│  │  NEIGHBORHOODS      │ │  POLICE PRECINCTS       ││
│  └─────────────────────┘ └──────────────────────────┘│
│  ┌─────────────────────┐ ┌──────────────────────────┐│
│  │  COUNCIL DISTRICTS  │ │  ZIP CODES              ││
│  └─────────────────────┘ └──────────────────────────┘│
│  ┌──────────────────────────────────────────────────┐│
│  │  TOP LOCATIONS (full width)                      ││
│  └──────────────────────────────────────────────────┘│
└──────────────────────────────────────────────────────┘
```

- Red/amber accent colors
- 2x2 grid of area breakdowns
- Locations full width below

## Analysis Page

- Pivot table with heatmap + raw data table
- Same dark theme styling
- Full width

## Styling

### Streamlit Theme (`.streamlit/config.toml`)

- Dark background (#0E1117 or similar)
- Dark secondary background for cards
- Accent color for interactive elements

### Plotly Charts

- Base template: `plotly_dark`
- Red/amber for geographic/severity data
- Blue/cyan for trend/temporal data
- Green/teal for positive metrics (clearances)
- Muted gridlines, consistent sizing

### Custom CSS (injected via `st.markdown`)

- KPI card styling: dark backgrounds, colored borders, glow effects
- Page title styling
- Sidebar customization
- No external CSS files — all inline

## Constraints

- Keep offense category as the sole sidebar filter
- Preserve all existing chart functionality and data processing
- Keep `utils.py` and `constants.py` structure intact
- Maintain 24h/1h/30m caching strategy
