"""
Agent 5: Chief Examiner — Synthesizes all reports into final assessment.

Features:
  - Cross-validates reports from all specialist agents
  - Detects inconsistencies between agent scores
  - Computes overall band score with evidence-based reasoning
  - Generates detailed Markdown feedback with explainability
  - Triggers self-correction loop via Critic when needed
"""

from __future__ import annotations

import re

from app.config import settings
from app.core.state import GraderState
from app.core.utils.llm import generate_json, llm_available, parse_json_object
from app.core.utils.rubric_loader import get_all_descriptions
from app.core.utils.prompt_templates import CHIEF_EXAMINER_PROMPT


# Agents that can act on a re-examination request (see critic.py).
RERUNNABLE_TARGETS = {"grounding", "coherence"}


def chief_examiner_node(state: GraderState) -> dict:
    """
    LangGraph node: Synthesize all agent reports into final assessment.

    Args:
        state: Current graph state with all agent outputs.

    Returns:
        State updates with overall_band, feedback, evidence_summary,
        and correction flags.
    """
    if llm_available():
        return _synthesize_with_llm(state)
    return _synthesize_rule_based(state)


def _synthesize_with_llm(state: GraderState) -> dict:
    """Use Gemini LLM for final synthesis and cross-validation."""
    try:
        # Load all rubric descriptions
        ta_rubric = get_all_descriptions("task_achievement")
        cc_rubric = get_all_descriptions("coherence_cohesion")
        lr_rubric = get_all_descriptions("lexical_resource")
        gra_rubric = get_all_descriptions("grammar_accuracy")

        ta_details = state.get("ta_details", {})
        cc_details = state.get("cc_details", {})
        sentence_metrics = state.get("sentence_metrics", {})
        lexical_metrics = state.get("lexical_metrics", {})
        limitations = _analysis_limitations(state)

        prompt = CHIEF_EXAMINER_PROMPT.format(
            ta_rubric=ta_rubric,
            cc_rubric=cc_rubric,
            lr_rubric=lr_rubric,
            gra_rubric=gra_rubric,
            ta_score=state.get("ta_score", 0),
            ta_confidence=state.get("ta_confidence", 0),
            coverage_rate=ta_details.get("coverage_rate", 0),
            contradiction_rate=ta_details.get("contradiction_rate", 0),
            missing_trends=ta_details.get("missing", 0),
            has_overview=ta_details.get("has_overview", False),
            gra_score=state.get("gra_score", 0),
            gra_confidence=state.get("gra_confidence", 0),
            lr_score=state.get("lr_score", 0),
            lr_confidence=state.get("lr_confidence", 0),
            grammar_error_count=len(state.get("grammar_errors", [])),
            error_types=sentence_metrics.get("error_type_counts", {}),
            complex_sentence_ratio=sentence_metrics.get("complex_sentence_ratio", 0),
            ttr=lexical_metrics.get("ttr", 0),
            academic_word_density=lexical_metrics.get("academic_word_density", 0),
            trend_word_repetition=lexical_metrics.get("trend_word_repetition", 0),
            cc_score=state.get("cc_score", 0),
            cc_confidence=state.get("cc_confidence", 0),
            cohesive_device_count=state.get("cohesion_report", {}).get(
                "cohesive_device_count", 0
            ),
            logical_progression=cc_details.get("logical_progression", "unknown"),
            paragraphing_quality=cc_details.get("paragraphing_quality", "unknown"),
            word_count=state.get("word_count", 0),
            analysis_notes="\n".join(f"- {note}" for note in limitations) or "None",
            task_prompt=(state.get("task_prompt") or "").strip() or "Not provided.",
            essay_text=state.get("essay_text", ""),
        )

        return _apply_chief_result(generate_json(prompt), state)

    except Exception as e:
        # Fallback to rule-based on LLM error
        result = _synthesize_rule_based(state)
        result["feedback"] += f"\n\n> ⚠️ LLM synthesis failed: {str(e)}. Using rule-based scoring."
        return result


def _analysis_limitations(state: GraderState) -> list[str]:
    """List the parts of the analysis that ran in a degraded mode."""
    notes = []
    if state.get("sentence_metrics", {}).get("language_tool_error"):
        notes.append(
            "Grammar and spelling error detection (LanguageTool) was unavailable, "
            "so GRA and LR rest on sentence-structure and vocabulary metrics only."
        )
    cc_details = state.get("cc_details", {})
    if cc_details and not cc_details.get("llm_assessed", True):
        notes.append(
            "The qualitative coherence assessment was unavailable, "
            "so CC rests on measured cohesion and paragraphing only."
        )
    return notes


def _overall_confidence(state: GraderState) -> float:
    """Average the specialist agents' confidence in their own evidence."""
    keys = ("ta_confidence", "cc_confidence", "lr_confidence", "gra_confidence")
    return round(sum(state.get(key) or 0.0 for key in keys) / len(keys), 3)


def _synthesize_rule_based(state: GraderState) -> dict:
    """Fallback: Compute final scores using simple averaging without LLM."""
    ta_score = state.get("ta_score", 0.0)
    cc_score = state.get("cc_score", 0.0)
    lr_score = state.get("lr_score", 0.0)
    gra_score = state.get("gra_score", 0.0)

    overall_confidence = _overall_confidence(state)

    # Build evidence summary
    ta_details = state.get("ta_details", {})
    lexical_metrics = state.get("lexical_metrics", {})
    sentence_metrics = state.get("sentence_metrics", {})

    evidence = {
        "ta_evidence": (
            f"Coverage {ta_details.get('coverage_rate', 0):.0%}, "
            f"Contradiction {ta_details.get('contradiction_rate', 0):.0%}, "
            f"Overview {'detected' if ta_details.get('has_overview') else 'missing'}"
        ),
        "cc_evidence": (
            f"Cohesive devices: {state.get('cohesion_report', {}).get('cohesive_device_count', 0)}, "
            f"Paragraphs: {state.get('cohesion_report', {}).get('paragraph_count', 0)}"
        ),
        "lr_evidence": (
            f"TTR {lexical_metrics.get('ttr', 0):.3f}, "
            f"Academic density {lexical_metrics.get('academic_word_density', 0):.3f}, "
            f"Trend repetition {lexical_metrics.get('trend_word_repetition', 0)}"
        ),
        "gra_evidence": (
            f"Complex ratio {sentence_metrics.get('complex_sentence_ratio', 0):.0%}, "
            f"Errors/100w {sentence_metrics.get('grammar_errors_per_100_words', 0):.1f}"
        ),
    }

    band_breakdown = {
        "TA": ta_score,
        "CC": cc_score,
        "LR": lr_score,
        "GRA": gra_score,
    }
    overall_band = _calculate_overall_band(band_breakdown)

    # Generate simple feedback
    feedback = _generate_rule_based_feedback(
        ta_score, cc_score, lr_score, gra_score,
        overall_band, evidence, state
    )

    score_attribution = _compute_score_attribution(state, band_breakdown)
    confidence_calibration = _compute_confidence_calibration(state, overall_confidence)

    return {
        "overall_band": overall_band,
        "overall_confidence": overall_confidence,
        "feedback": feedback,
        "band_breakdown": band_breakdown,
        "evidence_summary": evidence,
        "score_attribution": score_attribution,
        "confidence_calibration": confidence_calibration,
        "sentence_rewrites": state.get("sentence_rewrites", []),
        "analysis_warnings": _analysis_limitations(state),
        "needs_correction": False,
        "correction_target": "none",
        "critic_feedback": "",
    }


def _clamp_score(
    llm_score: float | None, agent_score: float, max_delta: float = 1.5
) -> float:
    """
    Clamp LLM-suggested score to be within ±max_delta of the agent's
    evidence-based score. This prevents the LLM from hallucinating scores
    that contradict quantitative evidence (e.g., giving TA=7.5 when
    NLI coverage is 0%).

    Args:
        llm_score: Score suggested by the LLM (may be None).
        agent_score: Evidence-based score from the specialist agent.
        max_delta: Maximum allowed deviation from agent score.

    Returns:
        Clamped score, rounded to nearest 0.5.
    """
    if llm_score is None:
        return agent_score

    clamped = max(agent_score - max_delta, min(agent_score + max_delta, llm_score))
    # Round to nearest 0.5
    return round(clamped * 2) / 2


def _calculate_overall_band(breakdown: dict[str, float]) -> float:
    """
    Compute the overall band score as average of criteria, capped if TA is extremely low.
    """
    ta = breakdown.get("TA", 0.0)
    raw_avg = sum(breakdown.values()) / len(breakdown)
    overall_band = round(raw_avg * 2) / 2
    
    # IELTS Capping Rule: If Task Achievement is extremely low (<= 2.0, e.g. due to completely
    # mismatched/off-topic response or non-attempt), cap overall score at TA + 2.0.
    if ta <= 2.0:
        overall_band = min(overall_band, ta + 2.0)
        
    return overall_band


def _parse_chief_response(raw_text: str, state: GraderState) -> dict:
    """Parse the Chief Examiner's JSON response."""
    try:
        result = parse_json_object(raw_text)
    except ValueError:
        return _synthesize_rule_based(state)
    return _apply_chief_result(result, state)


def _apply_chief_result(result: dict, state: GraderState) -> dict:
    """Validate the Chief Examiner's JSON verdict and turn it into state updates."""
    # Extract fields from LLM response
    final_scores = result.get("final_scores") or {}
    correction_target = str(result.get("correction_target") or "none").lower()
    needs_correction = (
        bool(result.get("needs_correction", False))
        and correction_target in RERUNNABLE_TARGETS
    )
    correction_count = state.get("correction_count", 0)

    # Prevent infinite loops
    if correction_count >= settings.max_correction_loops:
        needs_correction = False
    if not needs_correction:
        correction_target = "none"

    band_breakdown = {
        # TA is derived from chart-to-essay grounding. Keep this evidence-based
        # score authoritative instead of allowing the synthesis LLM to lower
        # or raise it based on free-form prose.
        "TA": state.get("ta_score", 0),
        "CC": _clamp_score(final_scores.get("cc"), state.get("cc_score", 0), max_delta=1.5),
        "LR": _clamp_score(final_scores.get("lr"), state.get("lr_score", 0), max_delta=1.5),
        "GRA": _clamp_score(final_scores.get("gra"), state.get("gra_score", 0), max_delta=1.5),
    }

    # The card scores are the source of truth. In particular, TA comes from
    # the grounding agent rather than the free-form synthesis model, so an
    # overall value calculated from the model's suggested TA can legitimately
    # differ from the displayed breakdown. Calculate it here from the scores
    # we actually return instead of discarding otherwise useful LLM feedback.
    overall_band = _calculate_overall_band(band_breakdown)
    feedback = result.get("feedback", "")
    if not isinstance(feedback, str) or not feedback.strip():
        return _synthesize_rule_based(state)

    # The synthesis model can use a slightly different score in its prose even
    # after its structured values have been clamped to the evidence-based card
    # scores. Preserve the useful qualitative feedback, but make every visible
    # score agree with the cards instead of needlessly falling back to rules.
    feedback = _normalize_chief_feedback(feedback, overall_band, band_breakdown)
    if not _chief_assessment_is_consistent(overall_band, band_breakdown, feedback):
        # Never show a candidate an LLM narrative that contradicts the JSON
        # scores displayed in the score cards.
        return _synthesize_rule_based(state)

    evidence_summary = result.get("evidence_summary")
    if not isinstance(evidence_summary, dict):
        evidence_summary = {}

    score_attribution = _compute_score_attribution(state, band_breakdown)
    # The model's self-reported confidence is not evidence; report the
    # specialist agents' own confidence instead.
    overall_conf = _overall_confidence(state)
    confidence_calibration = _compute_confidence_calibration(state, overall_conf)

    return {
        "overall_band": overall_band,
        "overall_confidence": overall_conf,
        "feedback": feedback,
        "band_breakdown": band_breakdown,
        "evidence_summary": evidence_summary,
        "score_attribution": score_attribution,
        "confidence_calibration": confidence_calibration,
        "sentence_rewrites": state.get("sentence_rewrites", []),
        "analysis_warnings": _analysis_limitations(state),
        "needs_correction": needs_correction,
        "correction_target": correction_target,
        "critic_feedback": result.get("correction_reason") or "",
    }


def _compute_score_attribution(state: GraderState, breakdown: dict[str, float]) -> dict:
    """
    List the measured strengths and weaknesses behind each criterion score.

    Every item quotes a metric the agents actually computed.  The band comes
    from threshold matching, not from adding points, so items carry a
    direction ("positive"/"negative") rather than a point value.
    """
    ta_details = state.get("ta_details", {})
    sentence_metrics = state.get("sentence_metrics", {})
    lexical_metrics = state.get("lexical_metrics", {})
    cohesion_report = state.get("cohesion_report", {})
    cc_details = state.get("cc_details", {})
    word_count = state.get("word_count", 0)

    def item(label: str, positive: bool) -> dict:
        return {"label": label, "type": "positive" if positive else "negative"}

    ta_items = []
    if ta_details.get("has_overview"):
        ta_items.append(item("Clear Overview statement detected", True))
    else:
        ta_items.append(item("Missing Overview statement", False))

    cov = ta_details.get("coverage_rate", 0)
    if cov >= 0.75:
        ta_items.append(item(f"High key feature coverage ({cov:.0%})", True))
    elif cov > 0:
        ta_items.append(item(f"Partial feature coverage ({cov:.0%})", False))
    else:
        ta_items.append(item("Key features omitted from chart", False))

    if ta_details.get("contradicted", 0) > 0:
        ta_items.append(item(
            f"Numeric inaccuracies detected ({ta_details['contradicted']} claim/s)", False
        ))

    if word_count < settings.min_word_count:
        ta_items.append(item(
            f"Word count below the {settings.min_word_count}-word minimum (-1.0 band)", False
        ))

    gra_items = []
    c_ratio = sentence_metrics.get("complex_sentence_ratio", 0)
    if c_ratio >= 0.4:
        gra_items.append(item(f"Good complex sentence ratio ({c_ratio:.0%})", True))
    else:
        gra_items.append(item(f"Low complex sentence ratio ({c_ratio:.0%})", False))

    if sentence_metrics.get("language_tool_error"):
        gra_items.append(item("Error density not measured (grammar checker unavailable)", False))
    else:
        errs = sentence_metrics.get("grammar_errors_per_100_words", 0)
        if errs <= 2.0:
            gra_items.append(item(f"Low error density ({errs:.1f}/100w)", True))
        else:
            gra_items.append(item(f"High error density ({errs:.1f}/100w)", False))

    lr_items = []
    ttr = lexical_metrics.get("ttr", 0)
    if ttr >= 0.5:
        lr_items.append(item(f"Good lexical diversity (TTR {ttr:.2f})", True))
    else:
        lr_items.append(item(f"Limited lexical diversity (TTR {ttr:.2f})", False))

    rep = lexical_metrics.get("trend_word_repetition", 0)
    if rep > 3:
        lr_items.append(item(f"Overused trend vocabulary ({rep} repetitions)", False))
    else:
        lr_items.append(item("Varied trend descriptors", True))

    cc_items = []
    if cohesion_report:
        devices = cohesion_report.get("cohesive_device_count", 0)
        cc_items.append(item(f"{devices} distinct cohesive devices used", devices >= 5))

        overused = cohesion_report.get("overused_devices", [])
        if overused:
            cc_items.append(item(f"Overused connectors: {', '.join(overused)}", False))

        paragraphs = cohesion_report.get("paragraph_count", 0)
        cc_items.append(item(f"{paragraphs} paragraph(s)", paragraphs >= 3))

    progression = cc_details.get("logical_progression", "unknown")
    if progression != "unknown":
        cc_items.append(item(f"Logical progression: {progression}", progression != "poor"))
    elif cc_details and not cc_details.get("llm_assessed", True):
        cc_items.append(item("Logical progression not assessed (LLM unavailable)", False))

    return {
        "TA": ta_items,
        "GRA": gra_items,
        "LR": lr_items,
        "CC": cc_items,
    }


def _compute_confidence_calibration(state: GraderState, overall_confidence: float) -> dict:
    """Report each specialist agent's own confidence next to the overall figure."""
    return {
        "grounding_nli": round(state.get("ta_confidence") or 0.0, 2),
        "grammar_eval": round(state.get("gra_confidence") or 0.0, 2),
        "lexical_eval": round(state.get("lr_confidence") or 0.0, 2),
        "coherence_eval": round(state.get("cc_confidence") or 0.0, 2),
        "overall": round(overall_confidence, 2),
    }


def _normalize_chief_feedback(feedback: str, overall_band: float, breakdown: dict) -> str:
    """Rewrite score labels in LLM prose to match the authoritative cards."""
    feedback = re.sub(
        r"(Overall\s+Band\s+Score\s*:\s*)[0-9]+(?:\.[0-9]+)?",
        rf"\g<1>{overall_band:.1f}",
        feedback,
        flags=re.IGNORECASE,
    )
    criterion_names = {
        "TA": "Task Achievement",
        "CC": "Coherence & Cohesion",
        "LR": "Lexical Resource",
        "GRA": "Grammatical Range & Accuracy",
    }
    for key, name in criterion_names.items():
        feedback = re.sub(
            rf"({re.escape(name)}[^\n]*?Band\s*)[0-9]+(?:\.[0-9]+)?",
            rf"\g<1>{float(breakdown[key]):.1f}",
            feedback,
            flags=re.IGNORECASE,
        )
    return feedback


def _chief_assessment_is_consistent(
    overall_band: float | str,
    breakdown: dict,
    feedback: str,
) -> bool:
    """Validate agreement between the LLM's structured score and prose."""
    try:
        scores = {key: float(value) for key, value in breakdown.items()}
        overall = float(overall_band)
    except (TypeError, ValueError):
        return False

    if any(score < 0 or score > 9 or score * 2 != round(score * 2) for score in scores.values()):
        return False

    expected_overall = _calculate_overall_band(scores)
    if overall != expected_overall:
        return False

    criterion_names = {
        "TA": "Task Achievement",
        "CC": "Coherence & Cohesion",
        "LR": "Lexical Resource",
        "GRA": "Grammatical Range & Accuracy",
    }
    for key, name in criterion_names.items():
        match = re.search(
            rf"{re.escape(name)}[^\n]*?Band\s*([0-9]+(?:\.5)?)",
            feedback,
            flags=re.IGNORECASE,
        )
        if match and float(match.group(1)) != scores[key]:
            return False

    # Validate that low scores (e.g. TA <= 4.0) do not contain contradicting praise in prose
    if scores.get("TA", 9.0) <= 4.0:
        ta_section_match = re.search(
            r"Task Achievement[^\n]*\n(.*?)(?=\n###|\Z)",
            feedback,
            flags=re.DOTALL | re.IGNORECASE,
        )
        if ta_section_match:
            ta_text = ta_section_match.group(1).lower()
            contradictory_phrases = [
                "successfully addresses all requirements",
                "covers all requirements",
                "addresses all requirements",
                "fully satisfies all",
                "accurately reported and supports",
            ]
            if any(phrase in ta_text for phrase in contradictory_phrases):
                return False

    return True


def _generate_rule_based_feedback(
    ta: float, cc: float, lr: float, gra: float,
    overall: float, evidence: dict, state: GraderState,
) -> str:
    """Generate Markdown feedback without LLM."""
    word_count = state.get("word_count", 0)
    errors = state.get("grammar_errors", [])
    limitations = "".join(
        f"\n> ⚠️ {note}" for note in _analysis_limitations(state)
    )

    feedback = f"""## 📊 IELTS Writing Task 1 Assessment

### Overall Band Score: {overall}

---

### Task Achievement — Band {ta}
{evidence['ta_evidence']}

### Coherence & Cohesion — Band {cc}
{evidence['cc_evidence']}

### Lexical Resource — Band {lr}
{evidence['lr_evidence']}

### Grammatical Range & Accuracy — Band {gra}
{evidence['gra_evidence']}

---

### 📝 Summary
- **Word count**: {word_count} words {'⚠️ (below 150 minimum!)' if word_count < settings.min_word_count else '✅'}
- **Grammar errors found**: {len(errors)}

> ℹ️ This assessment was generated using rule-based scoring.
> {'LLM feedback was unavailable or inconsistent with the score data, so this result uses evidence-based scoring.' if settings.gemini_api_key else 'For more detailed feedback, configure a Gemini API key.'}
"""
    return feedback + limitations
