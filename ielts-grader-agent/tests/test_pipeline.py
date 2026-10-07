"""End-to-end tests of the grading graph with every external service mocked."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]

CHART_DATA = {
    "chart_type": "bar",
    "title": "Renewable energy share in four countries",
    "categories": ["Australia", "Sweden", "Iceland", "Turkey"],
    "key_trends": [
        "Iceland's share increased from 46 in 1997 to 60 in 2000 and 70 in 2010.",
        "Turkey's share decreased from 37 in 1997 to 35 in 2010.",
    ],
}


class _NeutralNLI:
    def predict(self, pairs):
        # Strongly "neutral" logits: no entailment and no contradiction.
        return [[-3.0, -3.0, 3.0]] * len(pairs)


class _NoMatches:
    def check(self, text):
        return []


@pytest.fixture
def pipeline(monkeypatch, tmp_path):
    """Patch out Gemini, the NLI model and LanguageTool; return a call recorder."""
    from app.core.agents import (
        chart_analyzer, chief_examiner, coherence, critic, grammar_lexical, grounding,
    )

    calls = {"chart": 0, "chief": 0, "review": 0, "coherence": []}

    def analyze_chart(image_bytes, mime_type, task_prompt):
        calls["chart"] += 1
        return dict(CHART_DATA)

    def chief_json(prompt, **kwargs):
        calls["chief"] += 1
        # Ask for one re-examination, then accept the result.
        first = calls["chief"] == 1 and calls.get("request_correction", False)
        return {
            "needs_correction": first,
            "correction_target": "coherence" if first else None,
            "correction_reason": "CC looks inconsistent" if first else None,
            "final_scores": {"ta": 0.5, "cc": 6.0, "lr": 6.0, "gra": 6.0},
            "feedback": "## Assessment\n\nEvidence-based feedback.",
        }

    def coherence_json(prompt, **kwargs):
        calls["coherence"].append(prompt)
        return {"coherence_score": 6.0, "logical_progression": "good", "paragraphing_quality": "good"}

    def critic_json(prompt, **kwargs):
        return {
            "correction_target": "coherence",
            "specific_issue": "CC differs from the other criteria.",
            "suggested_focus": "Re-read paragraph grouping.",
            "severity": "medium",
        }

    for module in (chart_analyzer, chief_examiner, coherence, critic, grounding):
        monkeypatch.setattr(module, "llm_available", lambda: True)
    monkeypatch.setattr(chart_analyzer, "_analyze_with_gemini", analyze_chart)
    monkeypatch.setattr(chart_analyzer, "_CHART_CACHE", {})
    monkeypatch.setattr(chief_examiner, "generate_json", chief_json)
    monkeypatch.setattr(coherence, "generate_json", coherence_json)
    monkeypatch.setattr(critic, "generate_json", critic_json)
    monkeypatch.setattr(grounding, "get_nli_model", lambda: _NeutralNLI())
    monkeypatch.setattr(grammar_lexical, "get_language_tool", lambda: _NoMatches())

    image = tmp_path / "chart.png"
    image.write_bytes(b"not-a-real-image")
    calls["image_path"] = str(image)
    calls["essay"] = (PROJECT_ROOT / "data" / "samples" / "sample_essay_1.txt").read_text(
        encoding="utf-8"
    )
    return calls


def test_pipeline_grades_an_essay_end_to_end(pipeline):
    from app.core.graph import grade_essay

    finished = []
    result = grade_essay(
        pipeline["image_path"], pipeline["essay"], on_node_complete=finished.append
    )

    assert {"chart_analyzer", "grounding", "grammar_lexical", "coherence", "chief_examiner"} <= set(finished)
    assert finished.index("chart_analyzer") < finished.index("chief_examiner")
    assert "critic" not in finished

    breakdown = result["band_breakdown"]
    assert set(breakdown) == {"TA", "CC", "LR", "GRA"}
    # TA always comes from the grounding agent, never from the synthesis LLM.
    assert breakdown["TA"] == result["ta_score"] != 0.5
    # The introduction lists Iceland and the chart's years but none of its
    # values, so it must not count as reporting the Iceland trend.
    labels = {item["trend"][:7]: item["source"] for item in result["grounding_report"]}
    assert labels == {"Iceland": "subject", "Turkey'": "direct"}
    assert result["overall_band"] * 2 == round(result["overall_band"] * 2)
    assert result["feedback"].startswith("## Assessment")
    assert result["analysis_warnings"] == []
    assert result["correction_count"] == 0


def test_pipeline_caches_chart_analysis_per_image(pipeline):
    from app.core.graph import grade_essay

    first = grade_essay(pipeline["image_path"], pipeline["essay"])
    second = grade_essay(pipeline["image_path"], pipeline["essay"])

    assert pipeline["chart"] == 1
    assert first["ta_score"] == second["ta_score"]


def test_pipeline_correction_loop_passes_critic_feedback_to_rerun(pipeline):
    from app.core.graph import grade_essay

    pipeline["request_correction"] = True
    finished = []
    result = grade_essay(
        pipeline["image_path"], pipeline["essay"], on_node_complete=finished.append
    )

    assert "critic" in finished and "re_coherence" in finished
    assert result["correction_count"] == 1
    assert result["needs_correction"] is False
    assert pipeline["chief"] == 2
    # The first coherence call is the normal run; the re-run carries the concern.
    assert "Re-read paragraph grouping." not in pipeline["coherence"][0]
    assert "Re-read paragraph grouping." in pipeline["coherence"][1]


def test_pipeline_stops_cleanly_when_chart_cannot_be_read(pipeline, monkeypatch):
    from app.core.agents import chart_analyzer
    from app.core.graph import grade_essay

    def fail(image_bytes, mime_type, task_prompt):
        raise RuntimeError("unreadable image")

    monkeypatch.setattr(chart_analyzer, "_analyze_with_gemini", fail)
    result = grade_essay(pipeline["image_path"], pipeline["essay"])

    assert result["overall_band"] == 0.0
    assert "unreadable image" in result["feedback"]
    assert pipeline["chief"] == 0
