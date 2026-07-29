"""Smoke tests for deterministic utilities and safe degraded operation."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.utils.linguistics import (
    segment_sentences, compute_sentence_metrics,
    compute_lexical_metrics, resolve_coreferences, count_words,
)
from app.core.utils.cohesion_analyzer import analyze_cohesion, analyze_paragraph_structure
from app.core.utils.rubric_loader import match_score_to_band

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _sample_essay() -> str:
    return (PROJECT_ROOT / "data" / "samples" / "sample_essay_1.txt").read_text(
        encoding="utf-8"
    )


def test_core_utility_smoke():
    essay = _sample_essay()
    assert count_words(essay) >= 150

    sentences = segment_sentences(essay)
    assert len(sentences) >= 3
    assert len(resolve_coreferences(sentences)) == len(sentences)

    sentence_metrics = compute_sentence_metrics(essay)
    lexical_metrics = compute_lexical_metrics(essay)
    cohesion = analyze_cohesion(essay)
    paragraphs = analyze_paragraph_structure(essay)

    assert sentence_metrics["total_sentences"] == len(sentences)
    assert 0 <= lexical_metrics["ttr"] <= 1
    assert cohesion["cohesive_device_count"] > 0
    assert paragraphs["paragraph_count"] >= 2


def test_rubric_matching_returns_valid_bands():
    assert match_score_to_band("grammar_accuracy", {
        "complex_sentence_ratio": 0.7,
        "grammar_errors_per_100_words": 2.0,
        "sentence_variety_score": 0.6,
    }) in {float(i) for i in range(1, 10)}


def test_grammar_agent_degrades_gracefully_without_languagetool(monkeypatch):
    from app.core.agents import grammar_lexical

    def unavailable():
        raise RuntimeError("LanguageTool unavailable")

    monkeypatch.setattr(grammar_lexical, "get_language_tool", unavailable)
    result = grammar_lexical.grammar_lexical_node({
        "essay_text": "The figures show a steady increase over time.",
        "word_count": 8,
    })
    assert result["grammar_errors"] == []
    assert result["sentence_metrics"]["language_tool_error"] == "LanguageTool unavailable"


def test_graph_compiles():
    from app.core.graph import compile_graph
    assert compile_graph() is not None


def test_grounding_normalizes_logits_and_filters_wrong_subjects():
    from app.core.agents.grounding import (
        _is_relevant_contradiction,
        _mentions_trend_subject,
        _softmax,
    )

    probabilities = _softmax([6.0, 4.0, 0.0])
    assert round(sum(probabilities), 8) == 1.0
    assert 0 < probabilities[0] < 1
    assert not _is_relevant_contradiction(
        "Australia's value decreased from 9 to 5.",
        "Sweden increased over the same period.",
    )
    assert _is_relevant_contradiction(
        "Australia's value decreased from 9 to 5.",
        "Australia increased from 9 to 10.",
    )
    assert _mentions_trend_subject(
        "Australia's value decreased from 9 to 5.",
        "Australia decreased slightly by 2010.",
    )
    assert not _is_relevant_contradiction(
        "The percentage of commuters using trains increased from 20 to 40.",
        "The average cost of housing fell during the same period.",
    )
    assert _is_relevant_contradiction(
        "The percentage of commuters using trains increased from 20 to 40.",
        "The percentage of commuters using trains fell to 10.",
    )


def test_chief_rejects_conflicting_scores_and_feedback():
    from app.core.agents.chief_examiner import _chief_assessment_is_consistent

    breakdown = {"TA": 2.5, "CC": 6.5, "LR": 6.0, "GRA": 7.0}
    assert not _chief_assessment_is_consistent(
        6.5, breakdown, "### Task Achievement (Band 7.0)"
    )
    assert _chief_assessment_is_consistent(
        5.5, breakdown, "### Task Achievement (Band 2.5)"
    )


def test_chief_uses_authoritative_breakdown_when_llm_overall_differs():
    from app.core.agents.chief_examiner import _parse_chief_response

    state = {
        "ta_score": 1.0,
        "cc_score": 7.0,
        "lr_score": 4.0,
        "gra_score": 5.0,
        "correction_count": 0,
    }
    response = '''{
        "final_scores": {"ta": 2.0, "cc": 7.0, "lr": 4.0, "gra": 5.0},
        "overall_band": 4.5,
        "overall_confidence": 0.8,
        "feedback": "### Task Achievement (Band 1.0)"
    }'''

    result = _parse_chief_response(response, state)

    assert result["band_breakdown"]["TA"] == 1.0
    assert result["overall_band"] == 3.0


def test_chief_normalizes_mismatched_llm_prose_instead_of_falling_back():
    from app.core.agents.chief_examiner import _parse_chief_response

    state = {
        "ta_score": 3.0,
        "cc_score": 7.0,
        "lr_score": 4.0,
        "gra_score": 5.0,
        "correction_count": 0,
    }
    response = '''{
        "final_scores": {"ta": 5.0, "cc": 8.0, "lr": 4.0, "gra": 5.0},
        "overall_band": 5.5,
        "overall_confidence": 0.8,
        "feedback": "### Overall Band Score: 5.5\\n### Task Achievement (Band 5.0)\\n### Coherence & Cohesion (Band 8.0)"
    }'''

    result = _parse_chief_response(response, state)

    assert "rule-based scoring" not in result["feedback"]
    assert "Overall Band Score: 5.0" in result["feedback"]
    assert "Task Achievement (Band 3.0)" in result["feedback"]
    assert "Coherence & Cohesion (Band 8.0)" in result["feedback"]


def test_grounding_generic_categories():
    from app.core.agents.grounding import (
        _extract_mentioned_categories,
        _find_direct_trend_match,
        _is_relevant_contradiction,
    )
    
    categories = ["Kindergarten", "Grades 1-2", "Grades 3-4"]
    
    # Test category extraction
    assert _extract_mentioned_categories("Kindergarten students rose steadily", categories) == {"Kindergarten"}
    assert _extract_mentioned_categories("1st and 2nd grade dipped", categories) == {"Grades 1-2"}
    assert _extract_mentioned_categories("grades 3-4 rose", categories) == {"Grades 3-4"}
    
    # Test contradiction relevance with category list
    assert not _is_relevant_contradiction(
        "Kindergarten students rose from 2.4% to 3.0%",
        "1st and 2nd grade dipped to 1.2%",
        categories
    )
    assert _is_relevant_contradiction(
        "Kindergarten students rose from 2.4% to 3.0%",
        "Kindergarten students fell to 1.2%",
        categories
    )

    direct_match = _find_direct_trend_match(
        "Grades 5-6 rose from 1.5 in 1999 to 2.6 in 2004.",
        ["The proportion of home schooled 5th and 6th graders initially started at 1.5%, then reached 2.6% in 2004."],
        ["Grades 5- 6"],
    )
    assert direct_match is not None
    assert _find_direct_trend_match(
        "Grades 5-6 rose from 1.5 in 1999 to 2.6 in 2004.",
        ["Students in grades 1-2 and 5-6 were the least represented in 1999 and 2004."],
        ["Grades 1- 2", "Grades 5- 6"],
    ) is None


def test_grounding_adaptive_thresholds_in_correction(monkeypatch):
    from app.core.agents import grounding
    from app.core.agents.grounding import grounding_node

    class MockNLIModel:
        def predict(self, pairs):
            # logit scores corresponding to probs: [0.1, 0.4, 0.5]
            # (entailment = 0.4)
            return [[-2.3, -0.9, -0.7]] * len(pairs)

    monkeypatch.setattr(grounding, "get_nli_model", lambda: MockNLIModel())

    # Mock state
    state = {
        "chart_data": {
            "categories": ["Iceland"],
            "key_trends": [
                "Iceland shows a consistent upward trend, increasing from 46 in 1997 to 70 in 2010."
            ]
        },
        "essay_text": "This is a sentence that doesn't direct match.",
        "word_count": 8,
        "correction_count": 0,
        "correction_target": "none"
    }

    # Under strict/normal mode, entailment score 0.4 < 0.5 threshold, so it remains neutral
    result_normal = grounding_node(state)
    assert result_normal["grounding_report"][0]["label"] == "neutral"

    # Under correction mode, entailment score 0.4 > 0.35 threshold, so it is classified as entailment
    state_correction = state.copy()
    state_correction["correction_count"] = 1
    state_correction["correction_target"] = "grounding"

    result_correction = grounding_node(state_correction)
    assert result_correction["grounding_report"][0]["label"] == "entailment"


def test_chief_rejects_praise_when_ta_is_low():
    from app.core.agents.chief_examiner import _chief_assessment_is_consistent

    breakdown = {"TA": 1.0, "CC": 6.5, "LR": 6.0, "GRA": 7.0}
    praising_feedback = (
        "### Task Achievement (Band 1.0)\n"
        "The response successfully addresses all requirements of the task. "
        "You provided a clear overview of the trends."
    )
    assert not _chief_assessment_is_consistent(5.0, breakdown, praising_feedback)


def test_explainability_features():
    from app.core.agents.chief_examiner import _compute_score_attribution, _compute_confidence_calibration
    from app.core.agents.grammar_lexical import _generate_sentence_rewrites

    state = {
        "ta_details": {"has_overview": True, "coverage_rate": 0.8, "contradicted": 1, "missing": 1},
        "sentence_metrics": {"complex_sentence_ratio": 0.5, "grammar_errors_per_100_words": 1.5},
        "lexical_metrics": {"ttr": 0.6, "trend_word_repetition": 4},
        "word_count": 170,
        "ta_confidence": 0.90,
        "gra_confidence": 0.95,
    }
    breakdown = {"TA": 7.0, "CC": 7.0, "LR": 6.5, "GRA": 7.5}

    attr = _compute_score_attribution(state, breakdown)
    assert "TA" in attr and "GRA" in attr and "LR" in attr
    assert any(item["type"] == "positive" for item in attr["TA"])

    calib = _compute_confidence_calibration(state, 0.88)
    assert calib["grounding_nli"] == 0.90
    assert calib["grammar_eval"] == 0.95

    essay = "Regarding train usage, the number of travelers experienced a decline before it rose."
    rewrites = _generate_sentence_rewrites(essay, [])
    assert len(rewrites) > 0
    assert rewrites[0]["original_phrase"] == "experienced a decline"
    assert "declined modestly" in rewrites[0]["suggested_replacement"]


