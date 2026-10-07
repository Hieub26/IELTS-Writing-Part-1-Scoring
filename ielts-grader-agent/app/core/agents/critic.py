"""
Critic / Reflection Agent — Quality assurance for the grading pipeline.

When the Chief Examiner detects inconsistencies, the Critic Agent:
1. Analyzes the specific inconsistency
2. Identifies which specialist agent needs re-examination
3. Provides targeted feedback for the re-run

This follows the LangGraph Critic/Reflection pattern instead of
having the Chief directly re-invoke agents.

Only the grounding (TA) and coherence (CC) agents can be re-examined: both
take the Critic's feedback into an LLM review.  The grammar/lexical agent is
fully deterministic, so re-running it could never change its result.
"""

from __future__ import annotations

import json

from app.config import settings
from app.core.state import GraderState
from app.core.utils.llm import generate_json, llm_available
from app.core.utils.prompt_templates import CRITIC_PROMPT


RERUNNABLE_TARGETS = {"grounding", "coherence"}


def critic_node(state: GraderState) -> dict:
    """
    LangGraph node: Analyze inconsistencies and provide targeted correction feedback.

    Args:
        state: Current graph state with all agent outputs and Chief's correction flag.

    Returns:
        State updates with refined correction_target and critic_feedback.
    """
    correction_count = state.get("correction_count", 0)

    # Safety: prevent infinite loops
    if correction_count >= settings.max_correction_loops:
        return {
            "needs_correction": False,
            "correction_target": "none",
            "critic_feedback": "Maximum correction loops reached.",
            "correction_count": correction_count,
        }

    # If LLM is available, use it for nuanced analysis
    if llm_available():
        result = _critic_with_llm(state)
    else:
        result = _critic_rule_based(state)

    # Count only the loops that actually send an agent back for a re-run.
    result["correction_count"] = correction_count + (1 if result["needs_correction"] else 0)

    return result


def _critic_with_llm(state: GraderState) -> dict:
    """Use LLM to analyze the inconsistency in detail."""
    try:
        ta_details = state.get("ta_details", {})
        sentence_metrics = state.get("sentence_metrics", {})
        lexical_metrics = state.get("lexical_metrics", {})

        prompt = CRITIC_PROMPT.format(
            chief_assessment=json.dumps({
                "overall_band": state.get("overall_band"),
                "band_breakdown": state.get("band_breakdown"),
                "needs_correction": state.get("needs_correction"),
                "correction_target": state.get("correction_target"),
                "critic_feedback": state.get("critic_feedback"),
            }, indent=2),
            ta_score=state.get("ta_score", 0),
            ta_confidence=state.get("ta_confidence", 0),
            coverage_rate=ta_details.get("coverage_rate", 0),
            contradiction_rate=ta_details.get("contradiction_rate", 0),
            gra_score=state.get("gra_score", 0),
            gra_confidence=state.get("gra_confidence", 0),
            grammar_errors_per_100=sentence_metrics.get("grammar_errors_per_100_words", 0),
            lr_score=state.get("lr_score", 0),
            lr_confidence=state.get("lr_confidence", 0),
            ttr=lexical_metrics.get("ttr", 0),
            cc_score=state.get("cc_score", 0),
            cc_confidence=state.get("cc_confidence", 0),
        )

        return _interpret_critic_response(generate_json(prompt, temperature=0.1))

    except Exception:
        return _critic_rule_based(state)


def _critic_rule_based(state: GraderState) -> dict:
    """Rule-based critic analysis when LLM is unavailable."""
    ta_score = state.get("ta_score", 0)
    gra_score = state.get("gra_score", 0)
    lr_score = state.get("lr_score", 0)
    cc_score = state.get("cc_score", 0)

    ta_conf = state.get("ta_confidence", 0)

    # Detect specific inconsistencies
    issues = []

    # Case 1: GRA very high but TA very low
    if gra_score >= 7.0 and ta_score <= 4.0:
        issues.append({
            "target": "grounding",
            "feedback": (
                "GRA score is high ({:.1f}) but TA is very low ({:.1f}). "
                "Check whether accurate statements were missed or mislabelled "
                "as contradictions, and whether the low coverage is genuine."
            ).format(gra_score, ta_score),
            "severity": "high",
        })

    # Case 2: Low confidence on the grounding agent
    if ta_conf < settings.min_confidence_threshold:
        issues.append({
            "target": "grounding",
            "feedback": f"TA confidence is low ({ta_conf:.2f}). Re-examine NLI predictions.",
            "severity": "medium",
        })

    # Case 3: CC score dramatically different from others
    avg_other = (ta_score + lr_score + gra_score) / 3
    if abs(cc_score - avg_other) > 2.5:
        issues.append({
            "target": "coherence",
            "feedback": (
                f"CC score ({cc_score}) differs significantly from average "
                f"of other criteria ({avg_other:.1f}). Re-examine coherence assessment."
            ),
            "severity": "medium",
        })

    if issues:
        # Pick the highest severity issue
        issues.sort(key=lambda x: {"high": 0, "medium": 1, "low": 2}[x["severity"]])
        top_issue = issues[0]
        return {
            "needs_correction": True,
            "correction_target": top_issue["target"],
            "critic_feedback": top_issue["feedback"],
        }

    return {
        "needs_correction": False,
        "correction_target": "none",
        "critic_feedback": "No significant inconsistencies detected.",
    }


def _interpret_critic_response(result: dict) -> dict:
    """Turn the Critic's JSON verdict into state updates."""
    target = str(result.get("correction_target") or "none").lower()
    severity = str(result.get("severity") or "low").lower()

    if target not in RERUNNABLE_TARGETS or severity == "low":
        return {
            "needs_correction": False,
            "correction_target": "none",
            "critic_feedback": result.get("specific_issue") or "No issues found",
        }

    return {
        "needs_correction": True,
        "correction_target": target,
        "critic_feedback": (
            f"{result.get('specific_issue', '')} "
            f"Focus: {result.get('suggested_focus', '')}"
        ).strip(),
    }
