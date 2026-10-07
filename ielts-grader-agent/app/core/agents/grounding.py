"""
Agent 2: Feature Grounding — Verifies essay content against chart data using NLI.

Features:
  - Coreference Resolution preprocessing (rule-based)
  - Cross-Encoder NLI (DeBERTa-v3) for entailment/contradiction/neutral
  - Sliding window with resolved context
  - LLM second opinion when the Critic requests a re-examination
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
    find_overview_sentence,
)
from app.core.utils.llm import generate_json, llm_available
from app.core.utils.prompt_templates import (
    GROUNDING_REVIEW_PROMPT,
    OVERVIEW_DETECTION_PROMPT,
)
from app.core.utils.rubric_loader import match_score_to_band


# NLI label mapping (DeBERTa-v3 outputs: contradiction=0, entailment=1, neutral=2)
NLI_LABELS = {0: "contradiction", 1: "entailment", 2: "neutral"}

# Minimum NLI probability for a sentence to support / contradict a trend.
ENTAILMENT_THRESHOLD = 0.5
CONTRADICTION_THRESHOLD = 0.7

# A direct match needs the trend's category plus this many of its values/years.
DIRECT_MATCH_MIN_NUMBERS = 3
# Absolute tolerance when comparing a chart value with the essay's figure.
NUMBER_MATCH_TOLERANCE = 0.11

# Labels the second-opinion reviewer may assign.
REVIEW_LABELS = {"entailment", "partial", "contradiction", "neutral"}

_NUMBER_PATTERN = re.compile(r"\d+(?:\.\d+)?")


@lru_cache(maxsize=1)
def get_nli_model() -> CrossEncoder:
    """Load and cache the Cross-Encoder NLI model."""
    return CrossEncoder(settings.nli_model_name)


def grounding_node(state: GraderState) -> dict:
    """
    LangGraph node: Verify essay content against extracted chart data.

    Uses Cross-Encoder NLI to classify each essay sentence against
    each key trend from the chart, with coreference resolution preprocessing.
    When the Critic sends the essay back for re-examination, the labels that
    were not established by a direct numeric match get an LLM second opinion.

    Args:
        state: Current graph state with chart_data and essay_text.

    Returns:
        State updates with ta_score, ta_confidence, ta_details, grounding_report.
    """
    chart_data = state.get("chart_data", {})
    categories = [str(category) for category in chart_data.get("categories") or []]
    essay_text = state.get("essay_text", "")
    word_count = state.get("word_count", count_words(essay_text))

    # Handle missing chart data
    if not chart_data or not chart_data.get("key_trends"):
        return {
            "ta_score": 0.0,
            "ta_confidence": 0.0,
            "ta_details": {"error": "No chart data available for grounding"},
            "grounding_report": [],
        }

    key_trends = [str(trend) for trend in chart_data.get("key_trends", [])]

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
            "source": "nli",
        }

        # Prefer deterministic grounding for Task 1. A candidate who names
        # the same chart category and supports the claim with its values/years
        # has supplied stronger evidence than a generic NLI label can provide.
        direct_match = _find_direct_trend_match(trend, raw_sentences, categories)
        if direct_match:
            trend_coverage[trend_idx] = {
                "trend": trend,
                "best_label": "entailment",
                "best_score": direct_match["score"],
                "best_sentence": direct_match["sentence"],
                "best_sentence_idx": direct_match["sentence_idx"],
                "source": "direct",
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
                if entail_score > ENTAILMENT_THRESHOLD and entail_score > best_match["best_score"]:
                    best_match["best_label"] = "entailment"
                    best_match["best_score"] = float(entail_score)
                    best_match["best_sentence"] = pair_metadata[i]["original_sentence"]
                    best_match["best_sentence_idx"] = pair_metadata[i]["sentence_idx"]

                elif (
                    contra_score > CONTRADICTION_THRESHOLD
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
                    best_match["source"] = "subject"
                    break

        trend_coverage[trend_idx] = best_match

    # Step 3b: Second opinion when the Critic asked for a re-examination.
    # The reviewer may move a label in either direction; thresholds are never
    # relaxed, so a re-run cannot inflate the score by itself.
    is_correction = (
        state.get("correction_target") == "grounding"
        and state.get("correction_count", 0) > 0
    )
    review = {"requested": is_correction, "applied": False, "changed": 0}
    if is_correction and llm_available():
        try:
            review["changed"] = _review_with_llm(
                trend_coverage, raw_sentences, state.get("critic_feedback", "")
            )
            review["applied"] = True
        except Exception as exc:
            review["error"] = str(exc)

    # Step 4: Build detailed grounding report with Numeric Diff
    missing_trends_detail = []
    for trend_idx, match in trend_coverage.items():
        numeric_diff = None
        if match["best_label"] == "contradiction":
            expected_nums = _numbers_in_text(match["trend"], categories)
            found_nums = _numbers_in_text(match["best_sentence"], categories) if match["best_sentence"] else []
            if expected_nums or found_nums:
                exp_fmt = ", ".join(f"{n:g}" for n in expected_nums)
                found_fmt = ", ".join(f"{n:g}" for n in found_nums)
                numeric_diff = {
                    "expected": expected_nums,
                    "found": found_nums,
                    "formatted": f"Expected: {exp_fmt} ➔ Found: {found_fmt}" if exp_fmt and found_fmt else "Data mismatch",
                }

        score = match["best_score"]
        item = {
            "trend": match["trend"],
            "label": match["best_label"],
            # None when the label came from the LLM review, which yields no probability.
            "confidence": round(score, 3) if score is not None else None,
            "matched_sentence": match["best_sentence"],
            "sentence_index": match["best_sentence_idx"],
            "numeric_diff": numeric_diff,
            "source": match["source"],
        }
        if match.get("review_reason"):
            item["review_reason"] = match["review_reason"]
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
    overview_sentence, overview_source = _find_overview(essay_text, raw_sentences)
    has_overview = overview_sentence is not None

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
    scored = [r["confidence"] for r in grounding_report if r["confidence"] is not None]
    ta_confidence = round(sum(scored) / len(scored), 3) if scored else 0.0

    ta_details = {
        "total_trends": total_trends,
        "entailed": entailed,
        "partial": partial,
        "contradicted": contradicted,
        "missing": missing,
        "coverage_rate": round(coverage_rate, 3),
        "contradiction_rate": round(contradiction_rate, 3),
        "has_overview": has_overview,
        "overview_sentence": overview_sentence or "",
        "overview_source": overview_source,
        "word_count": word_count,
        "word_count_penalty_applied": word_count < settings.min_word_count,
        "missing_trends_detail": missing_trends_detail,
        "review": review,
    }

    return {
        "ta_score": ta_score,
        "ta_confidence": ta_confidence,
        "ta_details": ta_details,
        "grounding_report": grounding_report,
    }


def _review_with_llm(
    trend_coverage: dict[int, dict], sentences: list[str], critic_feedback: str
) -> int:
    """
    Ask Gemini to re-judge the trends that lack a direct numeric match.

    A changed label is accepted only when it cites an existing essay sentence
    (or is "neutral"), so the reviewer cannot award coverage without evidence.
    Updates ``trend_coverage`` in place and returns the number of changed labels.
    """
    reviewable = {
        idx: match for idx, match in trend_coverage.items() if match["source"] != "direct"
    }
    if not reviewable:
        return 0

    prompt = GROUNDING_REVIEW_PROMPT.format(
        critic_feedback=critic_feedback or "No specific concern was given.",
        numbered_sentences="\n".join(
            f"{idx}: {sentence}" for idx, sentence in enumerate(sentences)
        ),
        numbered_trends="\n".join(
            f"{idx}: [{match['best_label']}] {match['trend']}"
            for idx, match in reviewable.items()
        ),
    )
    result = generate_json(prompt, temperature=0)

    changed = 0
    for item in result.get("reviews") or []:
        if not isinstance(item, dict):
            continue
        match = reviewable.get(item.get("trend_index"))
        label = item.get("label")
        if match is None or label not in REVIEW_LABELS or label == match["best_label"]:
            continue

        sentence_idx = item.get("sentence_index")
        if label == "neutral":
            sentence, sentence_idx = "", -1
        elif type(sentence_idx) is int and 0 <= sentence_idx < len(sentences):
            sentence = sentences[sentence_idx]
        else:
            continue

        match.update({
            "best_label": label,
            "best_score": None,
            "best_sentence": sentence,
            "best_sentence_idx": sentence_idx,
            "source": "llm_review",
            "review_reason": str(item.get("reason", "")),
        })
        changed += 1

    return changed


def _find_overview(essay_text: str, sentences: list[str]) -> tuple[str | None, str]:
    """
    Locate the essay's overview sentence and report how it was found.

    A sentence opened by a standard marker ("Overall, ...") is accepted
    directly.  Strong candidates often write an overview without one, so when
    no marker is present Gemini is asked to point at the overview sentence.
    """
    sentence = find_overview_sentence(essay_text)
    if sentence:
        return sentence, "marker"
    if not llm_available():
        return None, "none"

    try:
        result = generate_json(
            OVERVIEW_DETECTION_PROMPT.format(
                numbered_sentences="\n".join(
                    f"{idx}: {text}" for idx, text in enumerate(sentences)
                )
            ),
            temperature=0,
        )
    except Exception:
        return None, "none"

    index = result.get("overview_sentence_index")
    if type(index) is int and 0 <= index < len(sentences):
        return sentences[index], "llm"
    return None, "none"


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
    normalized_text = _normalize_category_text(text.lower())
    text_numbers = set(_NUMBER_PATTERN.findall(normalized_text))

    for cat in categories:
        normalized_cat = _normalize_category_text(str(cat).lower()).strip()
        if not normalized_cat:
            continue

        # Whole-label match, tolerating a plural ("car" vs "cars")
        if re.search(
            rf"(?<![a-z0-9]){re.escape(normalized_cat)}(?:e?s)?(?![a-z0-9])",
            normalized_text,
        ):
            mentioned.add(cat)
            continue

        words = [w for w in re.findall(r"[a-z]+", normalized_cat) if len(w) > 2]
        numbers = _NUMBER_PATTERN.findall(normalized_cat)

        # Numbered labels such as "Grades 1-2" or "Ages 15-24": the candidate
        # may write "1st and 2nd grade".  Require every number plus the
        # label's own words, so two labels sharing a word stay distinct.
        if numbers:
            if all(num in text_numbers for num in numbers) and all(
                _singular(word) in normalized_text for word in words
            ):
                mentioned.add(cat)
            continue

        # Word-based match for multi-word categories
        significant = [w for w in words if len(w) > 3]
        if significant and all(w in normalized_text for w in significant):
            mentioned.add(cat)

    return mentioned


def _singular(word: str) -> str:
    return word[:-1] if len(word) > 3 and word.endswith("s") else word


def _normalize_category_text(text: str) -> str:
    """Normalise ordinal and range variants emitted by charts and candidates."""
    text = re.sub(r"(\d+)(?:st|nd|rd|th)\b", r"\1", text)
    return re.sub(r"\s*-\s*", "-", text)


def _find_direct_trend_match(
    trend: str,
    sentences: list[str],
    categories: list[str],
    min_matches: int = DIRECT_MATCH_MIN_NUMBERS,
    tolerance: float = NUMBER_MATCH_TOLERANCE,
) -> dict | None:
    """Find a category-aligned sentence with enough values/years for a trend."""
    trend_categories = _extract_mentioned_categories(trend, categories)
    if not trend_categories:
        return None

    trend_numbers = _numbers_in_text(trend, categories)
    trend_values = [number for number in trend_numbers if not _is_year(number)]
    for sentence_idx, sentence in enumerate(sentences):
        if not (trend_categories & _extract_mentioned_categories(sentence, categories)):
            continue
        sentence_numbers = _numbers_in_text(sentence, categories)
        matched_numbers = _count_matching_numbers(
            trend_numbers, sentence_numbers, tolerance=tolerance
        )
        # Years alone prove nothing: an introduction that lists the chart's
        # countries and years would otherwise "match" every trend.
        if trend_values and not _count_matching_numbers(
            trend_values, sentence_numbers, tolerance=tolerance
        ):
            continue
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
    """Extract data values/years, excluding numbers that belong to category labels."""
    # Ordinals ("5th and 6th graders") name a category; they are never data.
    text = re.sub(r"\b\d+(?:st|nd|rd|th)\b", "", text.lower())
    text = re.sub(r"\s*-\s*", "-", text)

    for category in categories or []:
        label = _normalize_category_text(str(category).lower()).strip()
        # A purely numeric category (e.g. the year "1997") is itself evidence.
        if not label or _NUMBER_PATTERN.fullmatch(label):
            continue
        # Strip the full label, then its bare range ("grades 1-2 and 5-6").
        ranges = re.findall(r"\d+(?:\.\d+)?-\d+(?:\.\d+)?", label)
        for fragment in [label, *ranges]:
            text = text.replace(fragment, " ")

    return [float(value) for value in _NUMBER_PATTERN.findall(text)]


def _is_year(number: float) -> bool:
    return number.is_integer() and 1800 <= number <= 2100


def _count_matching_numbers(expected: list[float], observed: list[float], tolerance: float = NUMBER_MATCH_TOLERANCE) -> int:
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
