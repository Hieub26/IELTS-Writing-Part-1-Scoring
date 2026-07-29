"""
Essay Highlighter Component — Grammarly-like highlighting for grammar errors
and NLI grounding results.

Renders the essay text with colored highlights:
  🔴 Red: Grammar/spelling errors (from Agent 3)
  🟡 Yellow: Contradicted information (from Agent 2)
  🟢 Green: Correctly described trends (from Agent 2)

Hover tooltips show error details and suggestions.
"""

from __future__ import annotations

import html
import streamlit as st


def render_highlighted_essay(
    essay_text: str,
    grammar_errors: list[dict],
    grounding_report: list[dict],
):
    """
    Render the essay with Grammarly-like highlights.

    Args:
        essay_text: Original essay text.
        grammar_errors: List of error dicts with offset, length, message, suggestion.
        grounding_report: List of NLI results with matched_sentence, label.
    """
    if not essay_text:
        return

    st.markdown("### 📝 Essay Analysis")

    # Build a list of annotations sorted by offset
    annotations = []

    # Add grammar error annotations
    for err in grammar_errors:
        offset = err.get("offset", 0)
        length = err.get("length", 0)
        if length <= 0:
            continue
        annotations.append({
            "start": offset,
            "end": offset + length,
            "type": "error",
            "message": err.get("message", "Grammar error"),
            "suggestion": ", ".join(err.get("suggestion", [])) or "No suggestion",
            "category": err.get("error_type", "grammar"),
        })

    # Add grounding annotations (sentence-level)
    for grounding in grounding_report:
        sentence = grounding.get("matched_sentence", "")
        label = grounding.get("label", "neutral")
        trend = grounding.get("trend", "")

        if not sentence or label == "neutral":
            continue

        # Find the sentence in the essay
        start = essay_text.find(sentence)
        if start == -1:
            continue

        if label == "contradiction":
            num_diff = grounding.get("numeric_diff", {})
            diff_text = f" ({num_diff.get('formatted')})" if num_diff and num_diff.get("formatted") else ""
            annotations.append({
                "start": start,
                "end": start + len(sentence),
                "type": "contradiction",
                "message": f"⚠️ Numeric Discrepancy{diff_text}",
                "suggestion": f"Chart feature: {trend}",
                "category": "factual",
            })

    # Sort by start offset (reverse for non-overlapping insertion)
    annotations.sort(key=lambda a: a["start"])

    # Remove overlapping annotations (keep the first one)
    cleaned = []
    last_end = -1
    for ann in annotations:
        if ann["start"] >= last_end:
            cleaned.append(ann)
            last_end = ann["end"]

    # Build highlighted HTML
    highlighted_html = _build_highlighted_html(essay_text, cleaned)

    # Render with custom CSS
    st.markdown(
        f'<div class="essay-container">{highlighted_html}</div>',
        unsafe_allow_html=True,
    )

    # Legend
    st.markdown(
        """
        <div style="display: flex; gap: 1.5rem; margin-top: 0.75rem; font-size: 0.8rem; color: #94a3b8;">
            <span><span style="background: rgba(239,68,68,0.2); border-bottom: 2px solid #ef4444; padding: 1px 6px; border-radius: 3px;">Grammar Error</span></span>
            <span><span style="background: rgba(245,158,11,0.2); border-bottom: 2px solid #f59e0b; padding: 1px 6px; border-radius: 3px;">Incorrect Data</span></span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _build_highlighted_html(text: str, annotations: list[dict]) -> str:
    """Build HTML with highlight spans and tooltips."""
    if not annotations:
        return html.escape(text).replace("\n", "<br>")

    parts = []
    last_idx = 0

    for ann in annotations:
        start = ann["start"]
        end = ann["end"]

        # Add non-highlighted text before this annotation
        if start > last_idx:
            safe_text = html.escape(text[last_idx:start])
            parts.append(safe_text.replace("\n", "<br>"))

        # Add highlighted text with tooltip
        highlighted_text = html.escape(text[start:end])
        css_class = _get_highlight_class(ann["type"])
        tooltip_html = _build_tooltip(ann)

        parts.append(
            f'<span class="{css_class}">'
            f'{highlighted_text}'
            f'{tooltip_html}'
            f'</span>'
        )

        last_idx = end

    # Add remaining text
    if last_idx < len(text):
        safe_text = html.escape(text[last_idx:])
        parts.append(safe_text.replace("\n", "<br>"))

    return "".join(parts)


def _get_highlight_class(annotation_type: str) -> str:
    """Map annotation type to CSS class."""
    return {
        "error": "highlight-error",
        "contradiction": "highlight-contradiction",
        "entailment": "highlight-entailment",
    }.get(annotation_type, "")


def _build_tooltip(annotation: dict) -> str:
    """Build tooltip HTML for an annotation."""
    if annotation["type"] == "entailment":
        return ""  # No tooltip needed for correct data

    message = html.escape(annotation.get("message", ""))
    suggestion = html.escape(annotation.get("suggestion", ""))

    tooltip_content = f"<strong>{message}</strong>"
    if suggestion:
        tooltip_content += f"<br>💡 {suggestion}"

    return f'<span class="error-tooltip">{tooltip_content}</span>'
