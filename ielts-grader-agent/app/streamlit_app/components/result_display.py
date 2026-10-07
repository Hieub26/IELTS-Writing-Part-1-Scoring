"""
Result Display Component — Shows scores, evidence, and detailed feedback.
"""

from __future__ import annotations

import html

import streamlit as st


def render_results(result: dict):
    """
    Render the complete grading results.

    Args:
        result: Final GraderState from the grading pipeline.
    """
    overall_band = result.get("overall_band", 0)
    overall_confidence = result.get("overall_confidence", 0)
    band_breakdown = result.get("band_breakdown", {})
    evidence_summary = result.get("evidence_summary", {})
    feedback = result.get("feedback", "")
    correction_count = result.get("correction_count", 0)

    # ═══════════════════════════════════════════
    # Overall Band Score
    # ═══════════════════════════════════════════

    correction_label = (
        f" · {correction_count} correction(s) applied" if correction_count > 0 else ""
    )
    overall_html = (
        '<div class="overall-band">'
        '<div class="band-label">OVERALL BAND SCORE</div>'
        f'<div class="band-value">{overall_band}</div>'
        '<div class="band-label">'
        f'{_confidence_badge(overall_confidence)}{correction_label}'
        '</div></div>'
    )
    st.markdown(overall_html, unsafe_allow_html=True)

    # Never present a degraded analysis as a full assessment.
    for warning in result.get("analysis_warnings", []):
        st.warning(f"⚠️ {warning}")

    st.markdown("")  # Spacer

    # ═══════════════════════════════════════════
    # Individual Criterion Scores
    # ═══════════════════════════════════════════

    cols = st.columns(4, gap="medium")
    criteria = [
        ("TA", "Task Achievement", None),
        ("CC", "Coherence & Cohesion", result.get("cc_confidence", 0)),
        ("LR", "Lexical Resource", result.get("lr_confidence", 0)),
        ("GRA", "Grammar Range & Accuracy", result.get("gra_confidence", 0)),
    ]

    for i, (key, name, conf) in enumerate(criteria):
        score = band_breakdown.get(key, 0)
        metric_badge = (
            _evidence_coverage_badge(result.get("ta_details", {}).get("coverage_rate", 0))
            if key == "TA"
            else _confidence_badge(conf)
        )
        with cols[i]:
            st.markdown(
                f"""
                <div class="score-card">
                    <div class="criterion-name">{name}</div>
                    <div class="band-number">{score}</div>
                    {metric_badge}
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("")  # Spacer

    # ═══════════════════════════════════════════
    # Evidence Summary
    # ═══════════════════════════════════════════

    if evidence_summary:
        with st.expander("🔍 Evidence & Metrics", expanded=False):
            for criterion, evidence in evidence_summary.items():
                # Evidence text is written by the LLM; escape before embedding.
                label = html.escape(str(criterion).replace("_evidence", "").upper())
                evidence = html.escape(str(evidence))
                st.markdown(
                    f"""
                    <div class="evidence-item">
                        <span class="metric-name">{label}</span>
                        <span class="metric-value">{evidence}</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    # ═══════════════════════════════════════════
    # Detailed Feedback
    # ═══════════════════════════════════════════

    if feedback:
        st.markdown("---")
        st.markdown(feedback)


def _confidence_badge(confidence: float) -> str:
    """Generate HTML for a confidence badge."""
    if confidence >= 0.8:
        css_class = "confidence-high"
        icon = "🟢"
    elif confidence >= 0.6:
        css_class = "confidence-medium"
        icon = "🟡"
    else:
        css_class = "confidence-low"
        icon = "🔴"

    return (
        f'<span class="confidence-badge {css_class}">'
        f'{icon} {confidence:.0%} confidence'
        f'</span>'
    )


def _evidence_coverage_badge(coverage: float) -> str:
    """Show TA evidence coverage without implying it is model confidence."""
    if coverage >= 0.8:
        css_class, icon = "confidence-high", "🟢"
    elif coverage >= 0.6:
        css_class, icon = "confidence-medium", "🟡"
    else:
        css_class, icon = "confidence-low", "🔴"

    return (
        f'<span class="confidence-badge {css_class}">'
        f'{icon} {coverage:.0%} evidence coverage'
        f'</span>'
    )
