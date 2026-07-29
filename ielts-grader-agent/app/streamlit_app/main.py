"""
IELTS Multi-Agent Grader — Streamlit Application.

Premium dark-themed UI with:
  - Chart image upload + essay text input
  - Grammarly-like essay highlighting
  - Radar chart for 4 criteria
  - Evidence-based scoring with confidence indicators
"""

from __future__ import annotations

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
from pathlib import Path

# Add project root to path
project_root = str(Path(__file__).resolve().parents[2])
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import streamlit as st

# ── Page Configuration ──
st.set_page_config(
    page_title="IELTS Writing Task 1 Grader — Multi-Agent AI",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Load Custom CSS ──
css_path = Path(__file__).parent / "assets" / "style.css"
if css_path.exists():
    # CSS contains Unicode characters; Windows' default cp1252 codec cannot
    # decode them reliably, so always read the project asset as UTF-8.
    with open(css_path, encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# ── Imports ──
from app.streamlit_app.components.upload_section import render_upload_section
from app.streamlit_app.components.radar_chart import render_radar_chart
from app.streamlit_app.components.essay_highlighter import render_highlighted_essay
from app.streamlit_app.components.result_display import render_results
from app.streamlit_app.components.rewrite_suggestions import render_rewrite_suggestions
from app.core.graph import grade_essay


# ══════════════════════════════════════════════
# Header
# ══════════════════════════════════════════════

st.markdown(
    """
    <div style="text-align: center; padding: 1rem 0 2rem 0;">
        <h1 style="font-size: 2.5rem; margin-bottom: 0.25rem;">
            📝 IELTS Writing Task 1 Grader
        </h1>
        <p style="color: #94a3b8; font-size: 1.1rem; font-weight: 400;">
            Multi-Agent AI System · Powered by LangGraph + Gemini + NLI
        </p>
        <div style="display: flex; justify-content: center; gap: 1.5rem; margin-top: 0.75rem;">
            <span style="color: #64748b; font-size: 0.8rem;">
                🤖 5 Specialized Agents
            </span>
            <span style="color: #64748b; font-size: 0.8rem;">
                🔄 Self-Correction Loop
            </span>
            <span style="color: #64748b; font-size: 0.8rem;">
                📊 Evidence-Based Scoring
            </span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ══════════════════════════════════════════════
# Upload Section
# ══════════════════════════════════════════════

image_path, essay_text = render_upload_section()

# ══════════════════════════════════════════════
# Grade Button
# ══════════════════════════════════════════════

st.markdown("")  # Spacer

col_btn = st.columns([1, 2, 1])
with col_btn[1]:
    grade_button = st.button(
        "🚀 Grade My Essay",
        use_container_width=True,
        disabled=not (image_path and essay_text),
        type="primary",
    )

# ══════════════════════════════════════════════
# Grading Pipeline
# ══════════════════════════════════════════════

if grade_button and image_path and essay_text:
    st.markdown("---")

    # Show loading animation
    with st.status("🤖 Multi-Agent Grading Pipeline Running...", expanded=True) as status:
        st.write("📊 **Agent 1**: Analyzing chart image with VLM...")
        st.write("🔍 **Agent 2**: Verifying essay content (NLI)...")
        st.write("📝 **Agent 3**: Checking grammar & vocabulary...")
        st.write("🔗 **Agent 4**: Evaluating coherence & cohesion...")
        st.write("👨‍⚖️ **Agent 5**: Chief Examiner synthesizing...")

        try:
            result = grade_essay(image_path, essay_text)
            status.update(label="✅ Grading Complete!", state="complete")
        except Exception as e:
            status.update(label="❌ Grading Failed", state="error")
            st.error(f"An error occurred: {str(e)}")
            st.stop()

    # ══════════════════════════════════════════════
    # Results Display
    # ══════════════════════════════════════════════

    st.markdown("")

    # Two-column layout: Radar Chart + Scores
    col_chart, col_scores = st.columns([1, 2], gap="large")

    with col_chart:
        render_radar_chart(
            result.get("band_breakdown", {}),
            result.get("overall_band", 0),
        )

    with col_scores:
        render_results(result)

    # ══════════════════════════════════════════════
    # Essay Highlighting
    # ══════════════════════════════════════════════

    st.markdown("---")

    render_highlighted_essay(
        essay_text=essay_text,
        grammar_errors=result.get("grammar_errors", []),
        grounding_report=result.get("grounding_report", []),
    )

    # ══════════════════════════════════════════════
    # Possible Alternatives & Sentence Rewrites
    # ══════════════════════════════════════════════

    st.markdown("---")

    render_rewrite_suggestions(
        sentence_rewrites=result.get("sentence_rewrites", []),
    )

    # ══════════════════════════════════════════════
    # Technical Details (Expandable)
    # ══════════════════════════════════════════════

    with st.expander("🔧 Technical Details", expanded=False):
        tab1, tab2, tab3, tab4 = st.tabs([
            "Chart Data", "TA Details", "Grammar/Lexical", "CC Details"
        ])

        with tab1:
            st.json(result.get("chart_data", {}))

        with tab2:
            st.json(result.get("ta_details", {}))
            st.markdown("#### Grounding Report")
            for item in result.get("grounding_report", []):
                label = item.get("label", "neutral")
                icon = {"entailment": "✅", "contradiction": "❌", "neutral": "➖"}.get(label, "➖")
                st.markdown(
                    f"{icon} **{label.upper()}** ({item.get('confidence', 0):.0%}): "
                    f"*{item.get('trend', '')}*"
                )
                if item.get("matched_sentence"):
                    st.caption(f"→ \"{item['matched_sentence']}\"")

        with tab3:
            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown("##### Sentence Metrics")
                st.json(result.get("sentence_metrics", {}))
            with col_b:
                st.markdown("##### Lexical Metrics")
                st.json(result.get("lexical_metrics", {}))

        with tab4:
            st.json(result.get("cc_details", {}))
            st.markdown("##### Cohesion Report")
            st.json(result.get("cohesion_report", {}))

# ══════════════════════════════════════════════
# Footer
# ══════════════════════════════════════════════

st.markdown(
    """
    <div style="text-align: center; padding: 3rem 0 1rem 0; color: #475569; font-size: 0.75rem;">
        <p>⚕️ This tool is for educational purposes only. Scores are AI-generated estimates.</p>
        <p>Built with LangGraph · Google Gemini · DeBERTa-v3 NLI · spaCy · LanguageTool</p>
    </div>
    """,
    unsafe_allow_html=True,
)
