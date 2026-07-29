"""
Critic / Reflection Agent — Quality assurance for the grading pipeline.

When the Chief Examiner detects inconsistencies, the Critic Agent:
1. Analyzes the specific inconsistency
2. Identifies which specialist agent needs re-examination
3. Provides targeted feedback for the re-run

This follows the LangGraph Critic/Reflection pattern instead of
having the Chief directly re-invoke agents.
"""

from __future__ import annotations

import json
import re
import time

from google import genai
from google.genai import types

from app.config import settings
from app.core.state import GraderState
from app.core.utils.prompt_templates import CRITIC_PROMPT


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
    if settings.gemini_api_key:
        result = _critic_with_llm(state)
    else:
        result = _critic_rule_based(state)

    # Increment correction count
    result["correction_count"] = correction_count + 1

    return result


def _critic_with_llm(state: GraderState) -> dict:
    """Use LLM to analyze the inconsistency in detail."""
    try:
        client = genai.Client(api_key=settings.gemini_api_key)

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

        response = None
        for attempt in range(3):
            try:
                response = client.models.generate_content(
                    model=settings.gemini_model_llm,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        temperature=0.1,
                        max_output_tokens=512,
                    ),
                )
                break
            except Exception as exc:
                is_transient = "503" in str(exc) or "429" in str(exc) or "UNAVAILABLE" in str(exc).upper()
                if not is_transient or attempt == 2:
                    raise
                time.sleep(2 ** attempt)

        if response is None:
            raise ValueError("No response generated from Gemini")

        return _parse_critic_response(response.text)

    except Exception:
        return _critic_rule_based(state)


def _critic_rule_based(state: GraderState) -> dict:
    """Rule-based critic analysis when LLM is unavailable."""
    ta_score = state.get("ta_score", 0)
    gra_score = state.get("gra_score", 0)
    lr_score = state.get("lr_score", 0)
    cc_score = state.get("cc_score", 0)

    ta_conf = state.get("ta_confidence", 0)
    gra_conf = state.get("gra_confidence", 0)
    cc_conf = state.get("cc_confidence", 0)

    # Detect specific inconsistencies
    issues = []

    # Case 1: GRA very high but TA very low
    if gra_score >= 7.0 and ta_score <= 4.0:
        issues.append({
            "target": "grounding",
            "feedback": (
                "GRA score is high ({:.1f}) but TA is very low ({:.1f}). "
                "The grounding agent may be too strict in NLI classification. "
                "Re-examine with lower contradiction threshold."
            ).format(gra_score, ta_score),
            "severity": "high",
        })

    # Case 2: Low confidence on any agent
    if ta_conf < settings.min_confidence_threshold:
        issues.append({
            "target": "grounding",
            "feedback": f"TA confidence is low ({ta_conf:.2f}). Re-examine NLI predictions.",
            "severity": "medium",
        })

    if gra_conf < settings.min_confidence_threshold:
        issues.append({
            "target": "grammar",
            "feedback": f"GRA confidence is low ({gra_conf:.2f}). Re-examine grammar analysis.",
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


def _parse_critic_response(raw_text: str) -> dict:
    """Parse the Critic's JSON response."""
    text = raw_text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*\n?", "", text)
        text = re.sub(r"\n?```\s*$", "", text)

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1:
        text = text[start : end + 1]

    try:
        result = json.loads(text)
        target = result.get("correction_target", "none")
        severity = result.get("severity", "low")

        if target == "none" or severity == "low":
            return {
                "needs_correction": False,
                "correction_target": "none",
                "critic_feedback": result.get("specific_issue", "No issues found"),
            }

        return {
            "needs_correction": True,
            "correction_target": target,
            "critic_feedback": (
                f"{result.get('specific_issue', '')} "
                f"Focus: {result.get('suggested_focus', '')}"
            ).strip(),
        }

    except json.JSONDecodeError:
        return {
            "needs_correction": False,
            "correction_target": "none",
            "critic_feedback": "Failed to parse critic response",
        }
