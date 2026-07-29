"""
Agent 3: Grammar & Lexical Evaluator — Assesses GRA and LR criteria.

Features:
  - LanguageTool integration for grammar/spelling error detection
  - spaCy-based sentence complexity analysis
  - Lexical metrics (TTR, academic word density, trend word repetition)
  - Error position tracking for Grammarly-like highlighting
  - Confidence scoring based on metric reliability
"""

from __future__ import annotations

from functools import lru_cache
import re

import language_tool_python

from app.config import settings
from app.core.state import GraderState
from app.core.utils.linguistics import (
    compute_sentence_metrics,
    compute_lexical_metrics,
    count_words,
)
from app.core.utils.rubric_loader import match_score_to_band


@lru_cache(maxsize=1)
def get_language_tool() -> language_tool_python.LanguageTool:
    """Load and cache the LanguageTool instance."""
    return language_tool_python.LanguageTool("en-US")


def grammar_lexical_node(state: GraderState) -> dict:
    """
    LangGraph node: Evaluate grammar accuracy and lexical resource.

    Args:
        state: Current graph state with essay_text.

    Returns:
        State updates with gra_score, lr_score, grammar_errors,
        lexical_metrics, sentence_metrics, and confidence scores.
    """
    essay_text = state.get("essay_text", "")
    word_count = state.get("word_count", count_words(essay_text))

    if not essay_text.strip():
        return _empty_result()

    # ═══════════════════════════════════════════
    # Part 1: Grammar & Spelling Analysis (GRA)
    # ═══════════════════════════════════════════

    language_tool_error = ""
    try:
        tool = get_language_tool()
        matches = tool.check(essay_text)
    except Exception as exc:
        # Java or the local LanguageTool distribution may be unavailable.
        # Continue with the deterministic metrics instead of failing the whole
        # grading pipeline, and expose the degraded analysis to the UI.
        matches = []
        language_tool_error = str(exc)

    # Filter and categorize grammar errors
    grammar_errors = []
    error_type_counts = {}

    for match in matches:
        # Skip very minor style suggestions
        if match.rule_issue_type == "style" and match.category == "TYPOGRAPHY":
            continue

        matched_word = essay_text[match.offset : match.offset + match.error_length].strip()
        matched_word_lower = matched_word.lower()

        # Ignore false spelling/grammar errors for compound terms and placeholders
        if any(w in matched_word_lower for w in (
            "home school", "home tutor", "somecountry", "somecity", "town a", "town b"
        )):
            continue

        # Ignore dialect/style variants (American vs British English)
        if (
            match.category in ("COLOCATIONS", "AMERICAN_ENGLISH_STYLE")
            or "British English" in match.message
            or "American English" in match.message
        ):
            continue

        error = {
            "offset": match.offset,
            "length": match.error_length,
            "error_type": match.rule_issue_type,  # grammar, misspelling, typographical
            "category": match.category,
            "rule_id": match.rule_id,
            "message": match.message,
            "context": match.context,
            "suggestion": match.replacements[:3] if match.replacements else [],
            "sentence": _extract_sentence_context(essay_text, match.offset),
        }
        grammar_errors.append(error)

        # Count error types
        etype = match.rule_issue_type
        error_type_counts[etype] = error_type_counts.get(etype, 0) + 1

    # Compute grammar error density
    total_errors = len(grammar_errors)
    errors_per_100 = (total_errors / word_count * 100) if word_count else 0

    # Spelling errors specifically
    spelling_errors = sum(
        1 for e in grammar_errors if e["error_type"] == "misspelling"
    )

    # ═══════════════════════════════════════════
    # Part 2: Sentence Complexity Analysis (GRA)
    # ═══════════════════════════════════════════

    sentence_metrics = compute_sentence_metrics(essay_text, settings.spacy_model)

    # ═══════════════════════════════════════════
    # Part 3: Lexical Resource Analysis (LR)
    # ═══════════════════════════════════════════

    lexical_metrics = compute_lexical_metrics(essay_text, settings.spacy_model)
    # Update spelling_errors from LanguageTool (more accurate than spaCy)
    lexical_metrics["spelling_errors"] = spelling_errors

    # ═══════════════════════════════════════════
    # Part 4: Score Computation
    # ═══════════════════════════════════════════

    # GRA metrics for rubric matching
    gra_metrics = {
        "complex_sentence_ratio": sentence_metrics["complex_sentence_ratio"],
        "grammar_errors_per_100_words": errors_per_100,
        "sentence_variety_score": sentence_metrics["sentence_variety_score"],
    }
    gra_score = match_score_to_band("grammar_accuracy", gra_metrics)

    # LR metrics for rubric matching
    lr_metrics = {
        "ttr": lexical_metrics["ttr"],
        "academic_word_density": lexical_metrics["academic_word_density"],
        "spelling_errors": spelling_errors,
        "trend_word_repetition": lexical_metrics["trend_word_repetition"],
    }
    lr_score = match_score_to_band("lexical_resource", lr_metrics)

    # ═══════════════════════════════════════════
    # Part 5: Confidence Scoring
    # ═══════════════════════════════════════════

    # GRA confidence: higher when we have more sentences to analyze
    total_sents = sentence_metrics["total_sentences"]
    gra_confidence = min(0.95, 0.5 + (total_sents / 20))  # Max out at ~10 sentences

    # LR confidence: higher when word count is substantial
    lr_confidence = min(0.95, 0.5 + (word_count / 300))  # Max out at ~150 words

    # Add error density info to sentence_metrics for Chief Examiner
    sentence_metrics["grammar_error_count"] = total_errors
    sentence_metrics["grammar_errors_per_100_words"] = round(errors_per_100, 2)
    sentence_metrics["error_type_counts"] = error_type_counts
    if language_tool_error:
        sentence_metrics["language_tool_error"] = language_tool_error

    # Extract sentence-level rewrite suggestions (Possible Alternatives)
    sentence_rewrites = _generate_sentence_rewrites(essay_text, grammar_errors)

    return {
        "gra_score": gra_score,
        "gra_confidence": round(gra_confidence, 3),
        "lr_score": lr_score,
        "lr_confidence": round(lr_confidence, 3),
        "grammar_errors": grammar_errors,
        "lexical_metrics": lexical_metrics,
        "sentence_metrics": sentence_metrics,
        "sentence_rewrites": sentence_rewrites,
    }


def _generate_sentence_rewrites(essay_text: str, grammar_errors: list[dict]) -> list[dict]:
    """Generate structured, actionable sentence-level rewrite suggestions."""
    rewrites = []
    
    # 1. Add trend phrase alternatives (Vocabulary Variety)
    trend_replacements = [
        ("experienced a decline", "declined modestly", "Avoid template phrase; use strong action verb", 0.91),
        ("witnessed an upward trend", "climbed steadily", "Avoid generic template phrase; use active descriptor", 0.93),
        ("experienced a similar trend", "followed a similar trajectory", "Enhance lexical variety", 0.88),
        ("witnessed a slow decline", "declined gradually", "Use precise adverbial descriptor", 0.90),
        ("saw an increase", "rose markedly", "Use strong action verb", 0.89),
    ]

    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", essay_text) if s.strip()]
    for sent_idx, sentence in enumerate(sentences):
        sent_lower = sentence.lower()
        for orig, replacement, reason, conf in trend_replacements:
            if orig in sent_lower:
                rewrites.append({
                    "sentence_idx": sent_idx + 1,
                    "original_phrase": orig,
                    "suggested_replacement": replacement,
                    "confidence": conf,
                    "reason": reason,
                    "type": "vocabulary",
                })

    # 2. Add top grammar suggestions from LanguageTool
    for err in grammar_errors:
        suggestions = err.get("suggestion", [])
        if suggestions and err.get("length", 0) > 0:
            offset = err.get("offset", 0)
            length = err.get("length", 0)
            orig_text = essay_text[offset : offset + length]
            if orig_text.strip():
                rewrites.append({
                    "sentence_idx": _get_sentence_num_for_offset(essay_text, offset),
                    "original_phrase": orig_text,
                    "suggested_replacement": suggestions[0],
                    "confidence": 0.95,
                    "reason": err.get("message", "Grammar correction"),
                    "type": "grammar",
                })

    return rewrites[:6]  # Top 6 high-value suggestions


def _get_sentence_num_for_offset(text: str, offset: int) -> int:
    """Find 1-indexed sentence number for a given character offset."""
    prefix = text[:offset]
    return max(1, len(re.findall(r"[.!?]\s+", prefix)) + 1)


def _extract_sentence_context(text: str, offset: int, window: int = 60) -> str:
    """Extract a text snippet around an error offset for context display."""
    start = max(0, offset - window)
    end = min(len(text), offset + window)

    # Find sentence boundaries
    snippet = text[start:end]
    # Add ellipsis if we truncated
    if start > 0:
        snippet = "..." + snippet
    if end < len(text):
        snippet = snippet + "..."

    return snippet


def _empty_result() -> dict:
    """Return empty result when no essay text is provided."""
    return {
        "gra_score": 0.0,
        "gra_confidence": 0.0,
        "lr_score": 0.0,
        "lr_confidence": 0.0,
        "grammar_errors": [],
        "lexical_metrics": {},
        "sentence_metrics": {},
    }
