"""
Agent 2: Feature Grounding — Verifies essay content against chart data using NLI.

Features:
  - Coreference Resolution preprocessing (rule-based)
  - Cross-Encoder NLI (DeBERTa-v3) for entailment/contradiction/neutral
  - Sliding window with resolved context
  - Coverage & contradiction rate computation
  - Heuristic-based TA scoring with confidence
"""

from __future__ import annotations

from functools import lru_cache
import math
import re

from sentence_transformers import CrossEncoder

from app.config import settings
from app.core.state import GraderState
from app.core.utils.linguistics import (
    segment_sentences,
    resolve_coreferences,
    count_words,
)
from app.core.utils.rubric_loader import match_score_to_band


# NLI label mapping (DeBERTa-v3 outputs: contradiction=0, entailment=1, neutral=2)
NLI_LABELS = {0: "contradiction", 1: "entailment", 2: "neutral"}


@lru_cache(maxsize=1)
def get_nli_model() -> CrossEncoder:
    """Load and cache the Cross-Encoder NLI model."""
    return CrossEncoder(settings.nli_model_name)


def grounding_node(state: GraderState) -> dict:
    """
    LangGraph node: Verify essay content against extracted chart data.

    Uses Cross-Encoder NLI to classify each essay sentence against
    each key trend from the chart, with coreference resolution preprocessing.

    Args:
        state: Current graph state with chart_data and essay_text.

    Returns:
        State updates with ta_score, ta_confidence, ta_details, grounding_report.
    """
    chart_data = state.get("chart_data", {})
    categories = chart_data.get("categories", [])
    essay_text = state.get("essay_text", "")
    word_count = state.get("word_count", count_words(essay_text))
    critic_feedback = state.get("critic_feedback", "")

    # Handle missing chart data
    if not chart_data or not chart_data.get("key_trends"):
        return {
            "ta_score": 0.0,
            "ta_confidence": 0.0,
            "ta_details": {"error": "No chart data available for grounding"},
            "grounding_report": [],
        }

    key_trends = chart_data.get("key_trends", [])
    notable_features = chart_data.get("notable_features", [])

    # Step 1: Segment sentences
    raw_sentences = segment_sentences(essay_text, settings.spacy_model)

    if not raw_sentences:
        return {
            "ta_score": 0.0,
            "ta_confidence": 0.0,
            "ta_details": {"error": "No sentences found in essay"},
            "grounding_report": [],
        }

    # Step 2: Coreference Resolution — replace pronouns with referents
    resolved_sentences = resolve_coreferences(raw_sentences, settings.spacy_model)

    # Step 3: Run NLI for each (trend, sentence) pair
    # Determine if this is a self-correction run targeting grounding
    is_correction = (
        state.get("correction_target") == "grounding"
        and state.get("correction_count", 0) > 0
    )

    # Relax thresholds during self-correction to be less strict
    entail_threshold = 0.35 if is_correction else 0.5
    contra_threshold = 0.85 if is_correction else 0.7
    min_matches = 2 if is_correction else 3
    tolerance = 1.1 if is_correction else 0.11

    # NOTE on pair direction: NLI entailment = "premise implies hypothesis".
    # Essay sentence is the premise (more specific, e.g. "tripled from 20% to 60%")
    # Chart trend is the hypothesis (more general, e.g. "increased from 20% to 60%")
    # A specific claim entails a general description.
    # We also check the reverse direction for robustness and take the best match.
    nli_model = get_nli_model()
    grounding_report = []
    trend_coverage = {}  # trend_index -> best match info

    for trend_idx, trend in enumerate(key_trends):
        best_match = {
            "trend": trend,
            "best_label": "neutral",
            "best_score": 0.0,
            "best_sentence": "",
            "best_sentence_idx": -1,
        }

        # Prefer deterministic grounding for Task 1. A candidate who names
        # the same chart category and supports the claim with its values/years
        # has supplied stronger evidence than a generic NLI label can provide.
        direct_match = _find_direct_trend_match(trend, raw_sentences, categories, min_matches=min_matches, tolerance=tolerance)
        if direct_match:
            trend_coverage[trend_idx] = {
                "trend": trend,
                "best_label": "entailment",
                "best_score": direct_match["score"],
                "best_sentence": direct_match["sentence"],
                "best_sentence_idx": direct_match["sentence_idx"],
            }
            continue

        # Create pairs: use individual sentences (NOT sliding window)
        # to avoid exceeding DeBERTa's 512-token limit.
        # We check BOTH directions and take the better entailment.
        forward_pairs = []   # (essay_sentence, trend)
        reverse_pairs = []   # (trend, essay_sentence)
        pair_metadata = []

        for sent_idx, resolved_sent in enumerate(resolved_sentences):
            # Truncate very long sentences to ~400 chars to stay within token limits
            context = resolved_sent[:400]

            forward_pairs.append((context, trend))
            reverse_pairs.append((trend, context))
            pair_metadata.append({
                "sentence_idx": sent_idx,
                "original_sentence": raw_sentences[sent_idx],
                "resolved_sentence": resolved_sent,
            })

        # Batch predict both directions
        if forward_pairs:
            fwd_scores = nli_model.predict(forward_pairs)
            rev_scores = nli_model.predict(reverse_pairs)

            for i in range(len(forward_pairs)):
                # Get probabilities for both directions
                fwd_probs = _softmax(fwd_scores[i])
                rev_probs = _softmax(rev_scores[i])

                # Entailment score = max entailment from either direction
                fwd_entail = fwd_probs[1]  # index 1 = entailment
                rev_entail = rev_probs[1]
                entail_score = max(fwd_entail, rev_entail)

                # Contradiction score = max contradiction from either direction
                fwd_contra = fwd_probs[0]  # index 0 = contradiction
                rev_contra = rev_probs[0]
                contra_score = max(fwd_contra, rev_contra)

                # Determine label based on best scores
                if entail_score > entail_threshold and entail_score > best_match["best_score"]:
                    best_match["best_label"] = "entailment"
                    best_match["best_score"] = float(entail_score)
                    best_match["best_sentence"] = pair_metadata[i]["original_sentence"]
                    best_match["best_sentence_idx"] = pair_metadata[i]["sentence_idx"]

                elif (
                    contra_score > contra_threshold
                    and best_match["best_label"] != "entailment"
                    and _is_relevant_contradiction(
                        trend, pair_metadata[i]["original_sentence"], categories
                    )
                ):
                    best_match["best_label"] = "contradiction"
                    best_match["best_score"] = float(contra_score)
                    best_match["best_sentence"] = pair_metadata[i]["original_sentence"]
                    best_match["best_sentence_idx"] = pair_metadata[i]["sentence_idx"]

        # IELTS responses often summarize a country's direction without every
        # value/year. Give such a relevant, non-contradictory sentence partial
        # coverage instead of treating the whole trend as absent.
        if best_match["best_label"] == "neutral":
            for metadata in pair_metadata:
                if _mentions_trend_subject(trend, metadata["original_sentence"], categories):
                    best_match["best_label"] = "partial"
                    best_match["best_score"] = 0.5
                    best_match["best_sentence"] = metadata["original_sentence"]
                    best_match["best_sentence_idx"] = metadata["sentence_idx"]
                    break

        trend_coverage[trend_idx] = best_match

    # Step 4: Build detailed grounding report with Numeric Diff
    missing_trends_detail = []
    for trend_idx, match in trend_coverage.items():
        numeric_diff = None
        if match["best_label"] == "contradiction":
            expected_nums = _numbers_in_text(match["trend"], categories)
            found_nums = _numbers_in_text(match["best_sentence"], categories) if match["best_sentence"] else []
            if expected_nums or found_nums:
                exp_fmt = ", ".join(f"{n:g}%" if n <= 100 else f"{n:g}" for n in expected_nums)
                found_fmt = ", ".join(f"{n:g}%" if n <= 100 else f"{n:g}" for n in found_nums)
                numeric_diff = {
                    "expected": expected_nums,
                    "found": found_nums,
                    "formatted": f"Expected: {exp_fmt} ➔ Found: {found_fmt}" if exp_fmt and found_fmt else "Data mismatch",
                }

        item = {
            "trend": match["trend"],
            "label": match["best_label"],
            "confidence": round(match["best_score"], 3),
            "matched_sentence": match["best_sentence"],
            "sentence_index": match["best_sentence_idx"],
            "numeric_diff": numeric_diff,
        }
        grounding_report.append(item)

        if match["best_label"] == "neutral":
            missing_trends_detail.append({
                "trend": match["trend"],
                "reason": "This key feature was omitted from the candidate's essay.",
            })

    # Step 5: Compute TA metrics
    total_trends = len(key_trends)
    entailed = sum(1 for r in grounding_report if r["label"] == "entailment")
    partial = sum(1 for r in grounding_report if r["label"] == "partial")
    contradicted = sum(1 for r in grounding_report if r["label"] == "contradiction")
    missing = sum(1 for r in grounding_report if r["label"] == "neutral")

    coverage_rate = (entailed + 0.5 * partial) / total_trends if total_trends else 0.0
    contradiction_rate = contradicted / total_trends if total_trends else 0.0

    # Check for overview presence
    has_overview = _detect_overview(essay_text)

    # Step 6: Match metrics to TA band score
    ta_metrics = {
        "coverage_rate": coverage_rate,
        "contradiction_rate": contradiction_rate,
        "has_overview": has_overview,
    }
    ta_score = match_score_to_band("task_achievement", ta_metrics)

    # Word count penalty: below 150 words → subtract 1.0 from TA
    if word_count < settings.min_word_count:
        ta_score = max(0.0, ta_score - 1.0)

    # Step 7: Compute confidence
    avg_confidence = (
        sum(r["confidence"] for r in grounding_report) / len(grounding_report)
        if grounding_report
        else 0.0
    )
    ta_confidence = round(avg_confidence, 3)

    ta_details = {
        "total_trends": total_trends,
        "entailed": entailed,
        "partial": partial,
        "contradicted": contradicted,
        "missing": missing,
        "coverage_rate": round(coverage_rate, 3),
        "contradiction_rate": round(contradiction_rate, 3),
        "has_overview": has_overview,
        "word_count": word_count,
        "word_count_penalty_applied": word_count < settings.min_word_count,
        "missing_trends_detail": missing_trends_detail,
    }

    return {
        "ta_score": ta_score,
        "ta_confidence": ta_confidence,
        "ta_details": ta_details,
        "grounding_report": grounding_report,
    }


def _softmax(logits) -> list[float]:
    """Convert one model-logit vector into stable class probabilities."""
    values = [float(value) for value in logits]
    maximum = max(values)
    exponentials = [math.exp(value - maximum) for value in values]
    total = sum(exponentials)
    return [value / total for value in exponentials]


def _extract_mentioned_categories(text: str, categories: list[str]) -> set[str]:
    """Helper to detect which categories are mentioned in a text."""
    mentioned = set()
    text_lower = text.lower()
    normalized_text = _normalize_category_text(text_lower)
    
    for cat in categories:
        cat_lower = cat.lower()
        normalized_cat = _normalize_category_text(cat_lower)
        
        # Exact match
        if normalized_cat in normalized_text:
            mentioned.add(cat)
            continue
            
        # Number-based match for grade levels like "1-2"
        numbers_in_cat = re.findall(r'\d+', normalized_cat)
        if numbers_in_cat:
            if all(num in normalized_text for num in numbers_in_cat) and "grade" in normalized_text:
                mentioned.add(cat)
                continue
                
        # Word-based match for multi-word categories (excluding generic words)
        words = [w for w in re.split(r'\W+', normalized_cat) if len(w) > 3 and w not in ("grade", "grades", "group", "class", "value")]
        if words and all(w in normalized_text for w in words):
            mentioned.add(cat)
            
    return mentioned


def _normalize_category_text(text: str) -> str:
    """Normalise grade/category variants emitted by charts and candidates."""
    text = text.replace("grades", "grade").replace("graders", "grade")
    text = re.sub(r"(\d+)(?:st|nd|rd|th)\b", r"\1", text)
    return re.sub(r"\s*-\s*", "-", text)


def _find_direct_trend_match(
    trend: str, sentences: list[str], categories: list[str], min_matches: int = 3, tolerance: float = 0.11
) -> dict | None:
    """Find a category-aligned sentence with enough values/years for a trend."""
    trend_categories = _extract_mentioned_categories(trend, categories)
    if not trend_categories:
        return None

    trend_numbers = _numbers_in_text(trend, categories)
    for sentence_idx, sentence in enumerate(sentences):
        if not (trend_categories & _extract_mentioned_categories(sentence, categories)):
            continue
        matched_numbers = _count_matching_numbers(
            trend_numbers, _numbers_in_text(sentence, categories), tolerance=tolerance
        )
        # A category plus three pieces of numeric evidence (e.g. start/end
        # values and a year) is a direct, high-confidence Task 1 match.
        if matched_numbers >= min_matches:
            return {
                "score": 0.95,
                "sentence": sentence,
                "sentence_idx": sentence_idx,
            }
    return None


def _numbers_in_text(text: str, categories: list[str] | None = None) -> list[float]:
    """Extract data values/years, excluding grade labels such as ``5-6``."""
    if categories:
        for category in categories:
            # Gemini and candidates vary whitespace around a grade range.
            category_pattern = re.escape(_normalize_category_text(category))
            text = re.sub(
                category_pattern.replace(r"\-", r"\s*-\s*"),
                "",
                _normalize_category_text(text),
                flags=re.IGNORECASE,
            )
    text = re.sub(
        r"\b\d{1,2}(?:st|nd|rd|th)?\s*(?:-|and)\s*\d{1,2}(?:st|nd|rd|th)?\s*(?:grade|grader|graders)?\b",
        "",
        text,
        flags=re.IGNORECASE,
    )
    return [float(value) for value in re.findall(r"\d+(?:\.\d+)?", text)]


def _count_matching_numbers(expected: list[float], observed: list[float], tolerance: float = 0.11) -> int:
    """Count expected values with an exact or IELTS-style rounded match."""
    matched = 0
    remaining = list(observed)
    for value in expected:
        for index, candidate in enumerate(remaining):
            if abs(value - candidate) <= tolerance:
                matched += 1
                remaining.pop(index)
                break
    return matched


def _is_relevant_contradiction(trend: str, sentence: str, categories: list[str] | None = None) -> bool:
    """Reject NLI contradictions that refer to a different named subject or category.

    A claim about Sweden must not count as a contradiction of a trend about
    Australia merely because their values differ. Generic trends without a
    possessive proper-name subject retain the NLI decision.
    """
    if categories:
        trend_cats = _extract_mentioned_categories(trend, categories)
        sent_cats = _extract_mentioned_categories(sentence, categories)
        # A sentence that names a different chart category cannot contradict
        # this trend.  The previous implementation only checked this when
        # *both* texts contained a category, letting unrelated claims through
        # when the chart-analysis model omitted the category from a trend description.
        if trend_cats or sent_cats:
            return bool(trend_cats & sent_cats)

    subject = _trend_subject(trend)
    if subject:
        return bool(re.search(rf"\b{re.escape(subject)}\b", sentence.lower()))

    # NLI often assigns a high contradiction score to two unrelated numeric
    # statements.  For a generic trend, require at least two shared topic
    # words before treating that score as a real chart-to-essay contradiction.
    return len(_topic_words(trend) & _topic_words(sentence)) >= 2


def _topic_words(text: str) -> set[str]:
    """Extract non-generic words used to establish a chart-claim topic."""
    ignored = {
        "a", "an", "and", "are", "as", "at", "between", "by", "for",
        "from", "in", "is", "of", "on", "or", "the", "to", "was", "were",
        "with", "value", "values", "figure", "figures", "data", "chart",
        "graph", "increase", "increased", "decrease", "decreased", "rose",
        "risen", "fell", "fallen", "grew", "growth", "declined", "decline",
        "higher", "lower", "highest", "lowest", "over", "during", "than",
    }
    return {
        token for token in re.findall(r"[a-zA-Z]+", text.lower())
        if len(token) > 2 and token not in ignored
    }


def _mentions_trend_subject(trend: str, sentence: str, categories: list[str] | None = None) -> bool:
    """Return whether a sentence discusses the named subject or category of a trend."""
    if categories:
        trend_cats = _extract_mentioned_categories(trend, categories)
        sent_cats = _extract_mentioned_categories(sentence, categories)
        if trend_cats and sent_cats:
            return bool(trend_cats & sent_cats)

    subject = _trend_subject(trend)
    return bool(subject and re.search(rf"\b{re.escape(subject)}\b", sentence.lower()))


def _trend_subject(trend: str) -> str | None:
    """Extract a possessive proper-name subject such as ``Australia``."""
    subject_match = re.search(r"\b([A-Z][A-Za-z-]+)'s\b", trend)
    return subject_match.group(1).lower() if subject_match else None


def _detect_overview(text: str) -> bool:
    """Detect if the essay contains an overview/summary statement."""
    overview_markers = [
        "overall", "in general", "it is clear", "it can be seen",
        "to summarize", "to summarise", "in summary", "generally speaking",
        "on the whole", "it is evident", "it is noticeable",
        "the most striking", "the most notable",
    ]
    text_lower = text.lower()
    return any(marker in text_lower for marker in overview_markers)
