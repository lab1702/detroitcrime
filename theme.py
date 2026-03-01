"""Theme and styling for Detroit Crime Dashboard"""

import plotly.graph_objects as go
import streamlit as st

# Accent color palettes
BLUE_ACCENT = "#00d4ff"      # Trends, temporal data
RED_ACCENT = "#ff4b4b"       # Geographic hotspots, severity
AMBER_ACCENT = "#ffaa00"     # Secondary geographic
GREEN_ACCENT = "#00c853"     # Positive metrics (clearances)
TEAL_ACCENT = "#00bfa5"      # Secondary positive

# Dark backgrounds
CARD_BG = "#1a1f2e"
CARD_BORDER = "#2d3548"

CUSTOM_CSS = """
<style>
    /* KPI metric cards */
    div[data-testid="stMetric"] {
        background-color: #1a1f2e;
        border: 1px solid #2d3548;
        border-radius: 8px;
        padding: 16px 20px;
        box-shadow: 0 0 15px rgba(0, 212, 255, 0.05);
    }
    div[data-testid="stMetric"] label {
        color: #8899aa;
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    div[data-testid="stMetric"] [data-testid="stMetricValue"] {
        color: #e0e0e0;
        font-size: 1.8rem;
        font-weight: 700;
    }

    /* Page headers */
    h1 {
        color: #e0e0e0 !important;
        font-weight: 700 !important;
        letter-spacing: -0.02em !important;
    }
    h2, h3 {
        color: #c0c8d4 !important;
        font-weight: 600 !important;
    }

    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background-color: #0a0e14;
        border-right: 1px solid #1a1f2e;
    }

    /* Footer */
    .footer-text {
        color: #556677;
        font-size: 0.8rem;
        text-align: center;
        padding-top: 2rem;
        border-top: 1px solid #1a1f2e;
    }
</style>
"""


def inject_css():
    """Inject custom CSS into the page."""
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def apply_dark_plotly(fig: go.Figure, accent_color: str = BLUE_ACCENT) -> go.Figure:
    """Apply dark theme to a Plotly figure with an accent color.

    Args:
        fig: Plotly figure to style.
        accent_color: Primary accent color for the chart.

    Returns:
        The same figure with dark theme applied.
    """
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0E1117",
        plot_bgcolor="#0E1117",
        font=dict(color="#c0c8d4"),
        xaxis=dict(gridcolor="#1a1f2e", zerolinecolor="#1a1f2e"),
        yaxis=dict(gridcolor="#1a1f2e", zerolinecolor="#1a1f2e"),
        coloraxis_colorbar=dict(
            tickfont=dict(color="#c0c8d4"),
            title=dict(font=dict(color="#c0c8d4")),
        ),
        margin=dict(l=20, r=20, t=40, b=20),
    )
    # Apply accent color to bar traces
    for trace in fig.data:
        if hasattr(trace, "marker") and trace.type == "bar":
            if trace.marker.color is None:
                trace.marker.color = accent_color
    return fig
