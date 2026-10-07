"""
Agent 4: Coherence & Cohesion Evaluator — Assesses CC criterion.

Combines:
  1. Quantitative Cohesion Analysis (code/NLP):
     - Cohesive device detection & categorization
     - Referencing quality (pronoun clarity)
     - Paragraph structure analysis
  2. Qualitative Coherence Analysis (LLM):
     - Logical progression between sentences
     - Paragraphing quality assessment
     - Overall organization evaluation
"""

from __future__ import annotations

from app.config import settings
from app.core.state import GraderState
from app.core.utils.linguistics import segment_sentences
from app.core.utils.cohesion_analyzer import (
    analyze_cohesion,
    analyze_paragraph_structure,
    analyze_referencing,
)
from app.core.utils.rubric_loader import (
    get_all_descriptions,
    match_score_to_band,
)
from app.core.utils.llm import generate_json, llm_available
from app.core.utils.prompt_templates import CC_COHERENCE_PROMPT


def coherence_node(state: GraderState) -> dict:
    """
    LangGraph node: Evaluate Coherence & Cohesion.

    Runs quantitative cohesion analysis first, then sends results
    to LLM for qualitative coherence assessment.

    Args:
        state: Current graph state with essay_text.

    Returns:
        State updates with cc_score, cc_confidence, cohesion_report,
        coherence_report, cc_details.
    """
    essay_text = state.get("essay_text", "")

    if not essay_text.strip():
        return _empty_result()

    # ═══════════════════════════════════════════
    # Part 1: Quantitative Cohesion Analysis
    # ═══════════════════════════════════════════

    # 1a. Cohesive devices detection
    cohesion_data = analyze_cohesion(essay_text)

    # 1b. Paragraph structure
    paragraph_data = analyze_paragraph_structure(essay_text)

    # 1c. Referencing quality
    sentences = segment_sentences(essay_text, settings.spacy_model)
    referencing_data = analyze_referencing(sentences, settings.spacy_model)

    # Build cohesion report
    cohesion_report = {
        **cohesion_data,
        **paragraph_data,
        **referencing_data,
    }

    # ═══════════════════════════════════════════
    # Part 2: Qualitative Coherence Analysis (LLM)
    # ═══════════════════════════════════════════

    is_correction = (
        state.get("correction_target") == "coherence"
        and state.get("correction_count", 0) > 0
    )
    coherence_report = _assess_coherence_with_llm(
        essay_text=essay_text,
        cohesion_data=cohesion_data,
        paragraph_data=paragraph_data,
        critic_feedback=state.get("critic_feedback", "") if is_correction else "",
    )
    llm_assessed = coherence_report.get("coherence_score") is not None

    # ═══════════════════════════════════════════
    # Part 3: Combined CC Score
    # ═══════════════════════════════════════════

    # Compute quantitative CC score from heuristics
    cc_quant_metrics = {
        "cohesive_device_count": cohesion_data["cohesive_device_count"],
        "repeated_connectors": cohesion_data["repeated_connectors"],
        "paragraph_count": paragraph_data["paragraph_count"],
    }
    cc_quant_score = match_score_to_band("coherence_cohesion", cc_quant_metrics)
    cc_qual_score = coherence_report.get("coherence_score")

    if llm_assessed:
        # Weighted blend (60% quantitative, 40% qualitative): code-measurable
        # aspects stay grounded while the LLM adds nuance.
        blend_weights = {"quantitative": 0.6, "qualitative": 0.4}
        cc_score = cc_quant_score * 0.6 + cc_qual_score * 0.4
    else:
        # No qualitative judgement is available.  Score on the measured
        # cohesion alone rather than blending in an invented placeholder.
        blend_weights = {"quantitative": 1.0, "qualitative": 0.0}
        cc_score = cc_quant_score

    # Round to nearest 0.5
    cc_score = round(cc_score * 2) / 2

    # ═══════════════════════════════════════════
    # Part 4: Confidence Scoring
    # ═══════════════════════════════════════════

    # Heuristic, not a calibrated probability: counting devices only covers
    # the cohesion half of the criterion, so confidence is capped at a medium
    # level unless the qualitative assessment also ran.
    if llm_assessed:
        cc_confidence = 0.75
        if paragraph_data["paragraph_count"] >= 3:
            cc_confidence += 0.05
        if paragraph_data["has_overview"]:
            cc_confidence += 0.05
    else:
        cc_confidence = 0.45

    cc_details = {
        "quantitative_score": cc_quant_score,
        "qualitative_score": cc_qual_score,
        "blend_weights": blend_weights,
        "llm_assessed": llm_assessed,
        "llm_error": coherence_report.get("error", ""),
        "logical_progression": coherence_report.get("logical_progression", "unknown"),
        "paragraphing_quality": coherence_report.get("paragraphing_quality", "unknown"),
        "coherence_issues": coherence_report.get("coherence_issues", []),
        "coherence_strengths": coherence_report.get("coherence_strengths", []),
    }

    return {
        "cc_score": cc_score,
        "cc_confidence": round(cc_confidence, 3),
        "cohesion_report": cohesion_report,
        "coherence_report": coherence_report,
        "cc_details": cc_details,
    }


def _assess_coherence_with_llm(
    essay_text: str,
    cohesion_data: dict,
    paragraph_data: dict,
    critic_feedback: str = "",
) -> dict:
    """
    Use Gemini LLM to assess qualitative coherence aspects.

    When the LLM is unavailable or its answer is unusable, the returned report
    has ``coherence_score`` set to None so the caller scores on quantitative
    evidence alone.
    """
    if not llm_available():
        return _unassessed_report("Gemini API key is not configured")

    review_note = ""
    if critic_feedback:
        review_note = (
            "\n## RE-EXAMINATION REQUEST:\n"
            "A quality reviewer questioned the previous coherence assessment: "
            f"{critic_feedback}\n"
            "Re-read the essay with this concern in mind. Confirm or revise the "
            "assessment in either direction based on the essay itself.\n"
        )

    try:
        prompt = CC_COHERENCE_PROMPT.format(
            cc_rubric=get_all_descriptions("coherence_cohesion"),
            essay_text=essay_text,
            cohesive_device_count=cohesion_data["cohesive_device_count"],
            device_variety_score=cohesion_data["device_variety_score"],
            overused_devices=", ".join(cohesion_data["overused_devices"]) or "None",
            paragraph_count=paragraph_data["paragraph_count"],
            has_intro=paragraph_data["has_intro"],
            has_overview=paragraph_data["has_overview"],
            structure_score=paragraph_data["structure_score"],
            review_note=review_note,
        )
        return _validate_coherence_report(generate_json(prompt))
    except Exception as exc:
        return _unassessed_report(f"LLM error: {exc}")


def _validate_coherence_report(report: dict) -> dict:
    """Normalise the LLM report, rejecting a missing or out-of-range score."""
    score = report.get("coherence_score")
    if isinstance(score, bool) or not isinstance(score, (int, float)) or not 0 <= score <= 9:
        raise ValueError(f"invalid coherence_score: {score!r}")

    report["coherence_score"] = round(float(score) * 2) / 2
    for key in ("coherence_issues", "coherence_strengths"):
        if not isinstance(report.get(key), list):
            report[key] = []
    return report


def _unassessed_report(reason: str) -> dict:
    """Report used when no qualitative coherence assessment could be made."""
    return {
        "coherence_score": None,
        "logical_progression": "unknown",
        "paragraphing_quality": "unknown",
        "coherence_issues": [],
        "coherence_strengths": [],
        "reasoning": "",
        "error": reason,
    }


def _empty_result() -> dict:
    """Return empty result when no essay text is provided."""
    return {
        "cc_score": 0.0,
        "cc_confidence": 0.0,
        "cohesion_report": {},
        "coherence_report": {},
        "cc_details": {},
    }
