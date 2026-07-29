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

import json
import time

from google import genai
from google.genai import types

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

    coherence_report = _assess_coherence_with_llm(
        essay_text=essay_text,
        cohesion_data=cohesion_data,
        paragraph_data=paragraph_data,
    )

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

    # Get qualitative score from LLM
    cc_qual_score = coherence_report.get("coherence_score", cc_quant_score)

    # Final CC score: weighted blend (60% quantitative, 40% qualitative)
    # This ensures code-measurable aspects are grounded while LLM adds nuance
    cc_score = round(cc_quant_score * 0.6 + cc_qual_score * 0.4, 1)

    # Round to nearest 0.5
    cc_score = round(cc_score * 2) / 2

    # ═══════════════════════════════════════════
    # Part 4: Confidence Scoring
    # ═══════════════════════════════════════════

    # Confidence based on:
    # - Device detection is deterministic (high confidence)
    # - LLM assessment has inherent variability
    # - More paragraphs = more data = higher confidence
    quant_confidence = 0.90  # Deterministic analysis
    qual_confidence = 0.70   # LLM-based

    if paragraph_data["paragraph_count"] >= 3:
        qual_confidence += 0.05
    if paragraph_data["has_overview"]:
        qual_confidence += 0.05

    cc_confidence = round(quant_confidence * 0.6 + qual_confidence * 0.4, 3)

    cc_details = {
        "quantitative_score": cc_quant_score,
        "qualitative_score": cc_qual_score,
        "blend_weights": {"quantitative": 0.6, "qualitative": 0.4},
        "logical_progression": coherence_report.get("logical_progression", "unknown"),
        "paragraphing_quality": coherence_report.get("paragraphing_quality", "unknown"),
        "coherence_issues": coherence_report.get("coherence_issues", []),
        "coherence_strengths": coherence_report.get("coherence_strengths", []),
    }

    return {
        "cc_score": cc_score,
        "cc_confidence": cc_confidence,
        "cohesion_report": cohesion_report,
        "coherence_report": coherence_report,
        "cc_details": cc_details,
    }


def _assess_coherence_with_llm(
    essay_text: str,
    cohesion_data: dict,
    paragraph_data: dict,
) -> dict:
    """
    Use Gemini LLM to assess qualitative coherence aspects.
    Falls back to quantitative-only scoring if LLM is unavailable.
    """
    if not settings.gemini_api_key:
        return {
            "coherence_score": 5.0,
            "logical_progression": "unknown",
            "paragraphing_quality": "unknown",
            "coherence_issues": ["LLM unavailable — coherence not assessed"],
            "coherence_strengths": [],
            "reasoning": "Fallback: LLM API key not configured",
        }

    try:
        client = genai.Client(api_key=settings.gemini_api_key)

        # Inject rubric descriptions and metrics into prompt
        cc_rubric = get_all_descriptions("coherence_cohesion")

        prompt = CC_COHERENCE_PROMPT.format(
            cc_rubric=cc_rubric,
            essay_text=essay_text,
            cohesive_device_count=cohesion_data["cohesive_device_count"],
            device_variety_score=cohesion_data["device_variety_score"],
            overused_devices=", ".join(cohesion_data["overused_devices"]) or "None",
            paragraph_count=paragraph_data["paragraph_count"],
            has_intro=paragraph_data["has_intro"],
            has_overview=paragraph_data["has_overview"],
            structure_score=paragraph_data["structure_score"],
        )

        response = None
        for attempt in range(3):
            try:
                response = client.models.generate_content(
                    model=settings.gemini_model_llm,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        temperature=0.2,
                        max_output_tokens=1024,
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

        return _parse_coherence_response(response.text)

    except Exception as e:
        return {
            "coherence_score": 5.0,
            "logical_progression": "unknown",
            "paragraphing_quality": "unknown",
            "coherence_issues": [f"LLM error: {str(e)}"],
            "coherence_strengths": [],
            "reasoning": f"Fallback due to error: {str(e)}",
        }


def _parse_coherence_response(raw_text: str) -> dict:
    """Parse JSON response from the coherence LLM."""
    import re

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
        # Validate required fields
        if "coherence_score" not in result:
            result["coherence_score"] = 5.0
        return result
    except json.JSONDecodeError:
        return {
            "coherence_score": 5.0,
            "logical_progression": "unknown",
            "paragraphing_quality": "unknown",
            "coherence_issues": ["Failed to parse LLM response"],
            "coherence_strengths": [],
            "reasoning": "JSON parse error",
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
