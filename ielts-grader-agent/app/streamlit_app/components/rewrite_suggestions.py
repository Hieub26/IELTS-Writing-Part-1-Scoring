"""
Possible Alternatives & Actionable Rewrites Component.
Renders concrete sentence-level revision suggestions.
"""

from __future__ import annotations

import html

import streamlit as st


def render_rewrite_suggestions(sentence_rewrites: list[dict]):
    """Render Actionable Sentence Rewrites Card."""
    st.markdown("### 💡 Possible Alternatives & Revision Suggestions")

    if not sentence_rewrites:
        st.success("✨ Great work! No repetitive trend phrases or major errors detected.")
        return

    st.caption("Contextual sentence-level revision suggestions to elevate your Band score.")

    for item in sentence_rewrites:
        # These strings come from the candidate's essay and the grammar
        # checker, so escape them before embedding in HTML.
        sent_num = item.get("sentence_idx", 1)
        orig = html.escape(str(item.get("original_phrase", "")))
        suggested = html.escape(str(item.get("suggested_replacement", "")))
        reason = html.escape(str(item.get("reason", "Grammar & vocabulary enhancement")))
        item_type = item.get("type", "vocabulary")

        type_badge = "🎨 Vocabulary Variety" if item_type == "vocabulary" else "📝 Grammar Accuracy"
        badge_bg = "#1e1b4b" if item_type == "vocabulary" else "#064e3b"
        badge_color = "#a5b4fc" if item_type == "vocabulary" else "#6ee7b7"

        st.markdown(
            f"""
            <div style="background: #0f172a; border: 1px solid #334155; border-radius: 8px; padding: 0.9rem 1.1rem; margin-bottom: 0.75rem;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
                    <span style="font-weight: 600; font-size: 0.85rem; background: {badge_bg}; color: {badge_color}; padding: 0.2rem 0.6rem; border-radius: 4px;">
                        Sentence {sent_num} · {type_badge}
                    </span>
                </div>
                <div style="display: flex; gap: 0.8rem; align-items: center; background: #1e293b; padding: 0.6rem; border-radius: 6px; font-family: monospace;">
                    <span style="color: #ef4444; text-decoration: line-through;">"{orig}"</span>
                    <span style="color: #38bdf8; font-weight: 700;">➔</span>
                    <span style="color: #22c55e; font-weight: 700;">"{suggested}"</span>
                </div>
                <div style="font-size: 0.82rem; color: #94a3b8; margin-top: 0.4rem;">
                    💡 <em>Reason: {reason}</em>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
