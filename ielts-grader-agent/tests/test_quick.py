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
    }) in {i / 2 for i in range(2, 19)}


def test_rubric_awards_half_band_only_when_close_to_next_band():
    def ta_band(coverage, has_overview=True):
        return match_score_to_band("task_achievement", {
            "coverage_rate": coverage,
            "contradiction_rate": 0.0,
            "has_overview": has_overview,
        })

    # Band 6 needs 0.60 coverage, Band 7 needs 0.75.
    assert ta_band(0.60) == 6.0
    assert ta_band(0.70) == 6.5
    assert ta_band(0.75) == 7.0
    # Nothing covered must not be rounded up from Band 1.
    assert ta_band(0.0) == 1.0
    # A missing overview is a hard requirement of Band 6, not a near miss.
    assert ta_band(0.58, has_overview=False) == 5.0


def test_overview_detection_requires_an_overview_sentence():
    from app.core.utils.linguistics import find_overview_sentence

    assert find_overview_sentence(
        "The chart shows car sales. Overall, sales rose in every country over the period."
    ).startswith("Overall")
    # The word "overall" used as an adjective mid-sentence is not an overview.
    assert find_overview_sentence(
        "The chart shows that between 1990 and 2000 the overall number of cars sold was high."
    ) is None
    assert find_overview_sentence("Overall. Sales rose.") is None


def test_grounding_asks_llm_for_overview_only_without_marker(monkeypatch):
    from app.core.agents import grounding

    sentences = [
        "The table shows home schooling rates.",
        "From the start, kindergarten pupils were home schooled the most and stayed highest.",
    ]
    essay = " ".join(sentences)

    def unexpected(prompt, **kwargs):
        raise AssertionError("LLM must not be called when a marker is present")

    monkeypatch.setattr(grounding, "llm_available", lambda: True)
    monkeypatch.setattr(grounding, "generate_json", unexpected)
    marked = "Overall, kindergarten pupils were home schooled the most in every year."
    assert grounding._find_overview(marked, [marked]) == (marked, "marker")

    monkeypatch.setattr(
        grounding, "generate_json", lambda prompt, **kwargs: {"overview_sentence_index": 1}
    )
    assert grounding._find_overview(essay, sentences) == (sentences[1], "llm")

    # "No overview", an out-of-range index and an unavailable LLM all mean none.
    for answer in ({"overview_sentence_index": -1}, {"overview_sentence_index": 9}, {}):
        monkeypatch.setattr(grounding, "generate_json", lambda prompt, answer=answer, **kwargs: answer)
        assert grounding._find_overview(essay, sentences) == (None, "none")

    monkeypatch.setattr(grounding, "llm_available", lambda: False)
    assert grounding._find_overview(essay, sentences) == (None, "none")


def test_trend_word_repetition_counts_whole_words_once():
    from app.core.utils.linguistics import _count_trend_word_repetitions

    counts = _count_trend_word_repetitions("sales increased and then increased again")
    assert counts["group_counts"]["increase"] == 2


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
    # With no error detection the GRA score must not be reported as reliable.
    assert result["gra_confidence"] <= 0.3


def test_grammar_agent_ignores_spelling_flags_on_chart_terms():
    from app.core.agents.grammar_lexical import _chart_vocabulary, _is_chart_term

    terms, text = _chart_vocabulary({
        "title": "Home schooled students in SomeCountry",
        "categories": ["Kindergarten", "Grades 1-2"],
    })
    assert _is_chart_term("SomeCountry", terms, text, is_spelling=True)
    assert _is_chart_term("home schooled", terms, text, is_spelling=False)
    # A grammar (not spelling) flag on a chart word is still a real error.
    assert not _is_chart_term("students", terms, text, is_spelling=False)
    assert not _is_chart_term("recieve", terms, text, is_spelling=True)


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


def _iceland_state() -> dict:
    return {
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


class _BelowThresholdNLI:
    def predict(self, pairs):
        # logit scores corresponding to probs: [0.1, 0.4, 0.5]
        # (entailment = 0.4, below the 0.5 threshold)
        return [[-2.3, -0.9, -0.7]] * len(pairs)


def test_grounding_correction_does_not_relax_thresholds(monkeypatch):
    from app.core.agents import grounding

    monkeypatch.setattr(grounding, "get_nli_model", lambda: _BelowThresholdNLI())
    monkeypatch.setattr(grounding, "llm_available", lambda: False)

    assert grounding.grounding_node(_iceland_state())["grounding_report"][0]["label"] == "neutral"

    # A re-run without a reviewer must reproduce the first pass, not inflate it.
    state = {**_iceland_state(), "correction_count": 1, "correction_target": "grounding"}
    result = grounding.grounding_node(state)
    assert result["grounding_report"][0]["label"] == "neutral"
    assert result["ta_details"]["review"] == {"requested": True, "applied": False, "changed": 0}


def test_grounding_correction_applies_evidence_backed_review(monkeypatch):
    from app.core.agents import grounding

    monkeypatch.setattr(grounding, "get_nli_model", lambda: _BelowThresholdNLI())
    monkeypatch.setattr(grounding, "llm_available", lambda: True)
    state = {
        **_iceland_state(),
        "essay_text": "Renewable energy in Iceland climbed from under half to roughly seventy percent.",
        "correction_count": 1,
        "correction_target": "grounding",
        "critic_feedback": "Check whether paraphrased figures were missed.",
    }

    prompts = []

    def cited_review(prompt, **kwargs):
        prompts.append(prompt)
        return {"reviews": [
            {"trend_index": 0, "label": "entailment", "sentence_index": 0, "reason": "Paraphrased figures"}
        ]}

    monkeypatch.setattr(grounding, "generate_json", cited_review)
    result = grounding.grounding_node(state)
    report = result["grounding_report"][0]
    assert report["label"] == "entailment"
    assert report["source"] == "llm_review"
    assert report["confidence"] is None
    assert result["ta_details"]["review"]["changed"] == 1
    assert "Check whether paraphrased figures were missed." in prompts[0]

    # A new label that cites no essay sentence is rejected.
    monkeypatch.setattr(grounding, "generate_json", lambda prompt, **kwargs: {"reviews": [
        {"trend_index": 0, "label": "entailment", "sentence_index": 7, "reason": "No evidence"}
    ]})
    result = grounding.grounding_node(state)
    assert result["grounding_report"][0]["label"] == "partial"
    assert result["ta_details"]["review"]["changed"] == 0

    # The reviewer can also lower a label.
    monkeypatch.setattr(grounding, "generate_json", lambda prompt, **kwargs: {"reviews": [
        {"trend_index": 0, "label": "neutral", "sentence_index": -1, "reason": "Not addressed"}
    ]})
    assert grounding.grounding_node(state)["grounding_report"][0]["label"] == "neutral"


def test_coherence_scores_on_measured_cohesion_when_llm_unavailable(monkeypatch):
    from app.core.agents import coherence

    essay = _sample_essay()
    monkeypatch.setattr(coherence, "llm_available", lambda: False)
    unassessed = coherence.coherence_node({"essay_text": essay})

    assert unassessed["cc_details"]["llm_assessed"] is False
    assert unassessed["cc_details"]["qualitative_score"] is None
    assert unassessed["cc_score"] == unassessed["cc_details"]["quantitative_score"]
    assert unassessed["cc_confidence"] < 0.5

    # An unusable LLM answer is treated the same way instead of becoming a 5.0.
    monkeypatch.setattr(coherence, "llm_available", lambda: True)
    monkeypatch.setattr(coherence, "generate_json", lambda prompt, **kwargs: {"coherence_score": "high"})
    assert coherence.coherence_node({"essay_text": essay})["cc_details"]["llm_assessed"] is False

    monkeypatch.setattr(coherence, "generate_json", lambda prompt, **kwargs: {
        "coherence_score": 8.0, "logical_progression": "good", "paragraphing_quality": "good",
    })
    assessed = coherence.coherence_node({"essay_text": essay})
    assert assessed["cc_details"]["llm_assessed"] is True
    assert assessed["cc_details"]["blend_weights"] == {"quantitative": 0.6, "qualitative": 0.4}


def test_critic_only_requests_reruns_that_can_change_the_result(monkeypatch):
    from app.core.agents import critic

    monkeypatch.setattr(critic, "llm_available", lambda: True)
    monkeypatch.setattr(critic, "generate_json", lambda prompt, **kwargs: {
        "correction_target": "grammar", "specific_issue": "GRA looks high", "severity": "high",
    })
    result = critic.critic_node({"correction_count": 0})
    assert result["needs_correction"] is False
    assert result["correction_count"] == 0

    monkeypatch.setattr(critic, "generate_json", lambda prompt, **kwargs: {
        "correction_target": "grounding", "specific_issue": "Coverage looks low",
        "suggested_focus": "Check paraphrases", "severity": "high",
    })
    result = critic.critic_node({"correction_count": 0})
    assert result["needs_correction"] is True
    assert result["correction_target"] == "grounding"
    assert result["correction_count"] == 1


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
    # No cohesion evidence was supplied, so no CC strengths may be claimed.
    assert attr["CC"] == []

    state["cohesion_report"] = {
        "cohesive_device_count": 1, "overused_devices": ["however"], "paragraph_count": 1,
    }
    cc_items = _compute_score_attribution(state, breakdown)["CC"]
    assert cc_items and all(item["type"] == "negative" for item in cc_items)

    calib = _compute_confidence_calibration(state, 0.88)
    assert calib["grounding_nli"] == 0.90
    assert calib["grammar_eval"] == 0.95

    essay = "Regarding train usage, the number of travelers experienced a decline before it rose."
    rewrites = _generate_sentence_rewrites(essay, [])
    assert len(rewrites) > 0
    assert rewrites[0]["original_phrase"] == "experienced a decline"
    assert "declined modestly" in rewrites[0]["suggested_replacement"]
    assert "confidence" not in rewrites[0]
