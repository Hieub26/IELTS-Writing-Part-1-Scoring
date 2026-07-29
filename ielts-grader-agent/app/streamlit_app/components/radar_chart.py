"""
Radar Chart Component — Displays the 4 IELTS criteria scores.
Uses Plotly Scatterpolar for an interactive, animated radar chart.
"""

from __future__ import annotations

import streamlit as st
import plotly.graph_objects as go


def render_radar_chart(band_breakdown: dict, overall_band: float = 0.0):
    """
    Render an interactive radar chart showing the 4 IELTS criteria scores.

    Args:
        band_breakdown: {"TA": 7.0, "CC": 6.5, "LR": 7.0, "GRA": 6.5}
        overall_band: Overall band score for the title.
    """
    categories = ["Task Achievement", "Coherence &\nCohesion", "Lexical\nResource", "Grammar\nRange & Accuracy"]
    short_keys = ["TA", "CC", "LR", "GRA"]

    values = [band_breakdown.get(k, 0) for k in short_keys]
    # Close the polygon
    values_closed = values + [values[0]]
    categories_closed = categories + [categories[0]]

    # Color based on overall band
    if overall_band >= 7.0:
        line_color = "rgba(16, 185, 129, 0.9)"   # Green
        fill_color = "rgba(16, 185, 129, 0.15)"
    elif overall_band >= 5.5:
        line_color = "rgba(59, 130, 246, 0.9)"    # Blue
        fill_color = "rgba(59, 130, 246, 0.15)"
    elif overall_band >= 4.0:
        line_color = "rgba(245, 158, 11, 0.9)"    # Yellow
        fill_color = "rgba(245, 158, 11, 0.15)"
    else:
        line_color = "rgba(239, 68, 68, 0.9)"     # Red
        fill_color = "rgba(239, 68, 68, 0.15)"

    fig = go.Figure()

    # Score trace
    fig.add_trace(go.Scatterpolar(
        r=values_closed,
        theta=categories_closed,
        fill="toself",
        fillcolor=fill_color,
        line=dict(color=line_color, width=2.5),
        marker=dict(size=8, color=line_color),
        name="Your Score",
        hovertemplate="<b>%{theta}</b><br>Band: %{r:.1f}<extra></extra>",
    ))

    # Reference trace: Band 6 baseline
    baseline = [6] * 5
    fig.add_trace(go.Scatterpolar(
        r=baseline,
        theta=categories_closed,
        fill=None,
        line=dict(color="rgba(148, 163, 184, 0.3)", width=1, dash="dash"),
        name="Band 6 Baseline",
        hoverinfo="skip",
    ))

    fig.update_layout(
        polar=dict(
            bgcolor="rgba(0, 0, 0, 0)",
            radialaxis=dict(
                visible=True,
                range=[0, 9],
                tickvals=[0, 3, 5, 6, 7, 9],
                ticktext=["0", "3", "5", "6", "7", "9"],
                tickfont=dict(size=10, color="#64748b"),
                gridcolor="rgba(255, 255, 255, 0.06)",
                linecolor="rgba(255, 255, 255, 0.06)",
            ),
            angularaxis=dict(
                tickfont=dict(size=11, color="#94a3b8", family="Inter"),
                gridcolor="rgba(255, 255, 255, 0.06)",
                linecolor="rgba(255, 255, 255, 0.08)",
            ),
        ),
        showlegend=False,
        paper_bgcolor="rgba(0, 0, 0, 0)",
        plot_bgcolor="rgba(0, 0, 0, 0)",
        margin=dict(l=60, r=60, t=20, b=20),
        height=350,
    )

    st.plotly_chart(fig, use_container_width=True)
