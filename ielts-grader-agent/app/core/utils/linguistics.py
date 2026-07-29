"""
Linguistics Utilities — spaCy-based analysis for IELTS grading.
Provides:
  - Sentence segmentation
  - Complex sentence detection (advcl, relcl, ccomp, acl)
  - Sentence variety scoring
  - Coreference resolution (rule-based pronoun replacement)
  - Word counting
"""

from __future__ import annotations

import re
from collections import Counter
from functools import lru_cache

import spacy


@lru_cache(maxsize=1)
def get_nlp(model_name: str = "en_core_web_sm"):
    """Load and cache the spaCy model."""
    return spacy.load(model_name)


# ──────────────────────────────────────────────
# Sentence Segmentation
# ──────────────────────────────────────────────

def segment_sentences(text: str, model_name: str = "en_core_web_sm") -> list[str]:
    """Split text into sentences using spaCy."""
    nlp = get_nlp(model_name)
    doc = nlp(text)
    return [sent.text.strip() for sent in doc.sents if sent.text.strip()]


# ──────────────────────────────────────────────
# Complex Sentence Detection
# ──────────────────────────────────────────────

# Dependency labels that indicate subordinate/complex clause structures
COMPLEX_DEP_LABELS = {"advcl", "relcl", "ccomp", "acl", "xcomp"}


def is_complex_sentence(sentence_text: str, model_name: str = "en_core_web_sm") -> bool:
    """
    Determine if a sentence is grammatically complex.
    A sentence is complex if it contains at least one subordinate clause
    (advcl, relcl, ccomp, acl, xcomp).
    """
    nlp = get_nlp(model_name)
    doc = nlp(sentence_text)
    for token in doc:
        if token.dep_ in COMPLEX_DEP_LABELS:
            return True
    return False


def compute_sentence_metrics(text: str, model_name: str = "en_core_web_sm") -> dict:
    """
    Compute sentence-level metrics for Grammatical Range & Accuracy.

    Returns:
        {
            "total_sentences": int,
            "complex_sentences": int,
            "simple_sentences": int,
            "complex_sentence_ratio": float,
            "avg_sentence_length": float,
            "sentence_lengths": list[int],
            "sentence_variety_score": float,
        }
    """
    nlp = get_nlp(model_name)
    doc = nlp(text)
    sentences = [sent for sent in doc.sents if sent.text.strip()]

    if not sentences:
        return {
            "total_sentences": 0,
            "complex_sentences": 0,
            "simple_sentences": 0,
            "complex_sentence_ratio": 0.0,
            "avg_sentence_length": 0.0,
            "sentence_lengths": [],
            "sentence_variety_score": 0.0,
        }

    complex_count = 0
    sentence_lengths = []

    for sent in sentences:
        length = len([t for t in sent if not t.is_punct and not t.is_space])
        sentence_lengths.append(length)

        # Check for complex structure
        deps_in_sent = {token.dep_ for token in sent}
        if deps_in_sent & COMPLEX_DEP_LABELS:
            complex_count += 1

    total = len(sentences)
    simple_count = total - complex_count
    avg_length = sum(sentence_lengths) / total if total else 0

    # Sentence variety score: measures how diverse sentence lengths are
    # Normalized standard deviation of sentence lengths (0-1)
    variety_score = _compute_variety_score(sentence_lengths)

    return {
        "total_sentences": total,
        "complex_sentences": complex_count,
        "simple_sentences": simple_count,
        "complex_sentence_ratio": complex_count / total if total else 0.0,
        "avg_sentence_length": round(avg_length, 1),
        "sentence_lengths": sentence_lengths,
        "sentence_variety_score": round(variety_score, 3),
    }


def _compute_variety_score(lengths: list[int]) -> float:
    """
    Compute a sentence variety score based on coefficient of variation.
    Higher score = more variety in sentence lengths = better writing.
    Capped at 1.0.
    """
    if len(lengths) < 2:
        return 0.0

    mean = sum(lengths) / len(lengths)
    if mean == 0:
        return 0.0

    variance = sum((x - mean) ** 2 for x in lengths) / len(lengths)
    std_dev = variance ** 0.5
    cv = std_dev / mean  # Coefficient of variation

    # Normalize: CV of 0.5+ maps to score 1.0
    return min(cv / 0.5, 1.0)


# ──────────────────────────────────────────────
# Coreference Resolution (Rule-based)
# ──────────────────────────────────────────────

# Pronouns to resolve
PRONOUNS_TO_RESOLVE = {"it", "they", "this", "these", "its", "their", "them"}

# Idiomatic expressions where 'it' should NOT be resolved
# These are common impersonal constructions in IELTS Task 1
IDIOMATIC_IT_PATTERNS = {
    "it is", "it was", "it can", "it could", "it may", "it might",
    "it will", "it would", "it should", "it has", "it had",
    "it seems", "it appears", "it remains",
}


def resolve_coreferences(
    sentences: list[str], model_name: str = "en_core_web_sm"
) -> list[str]:
    """
    Simple rule-based coreference resolution for IELTS essays.
    Replaces pronouns (it, they, this, these) with the most likely
    noun phrase from the preceding sentence.

    This is designed for IELTS Task 1 essays where "it" typically refers to
    a data trend, "they" to multiple categories, etc.

    Args:
        sentences: List of sentences in order.

    Returns:
        List of sentences with pronouns replaced by referent noun phrases.
    """
    nlp = get_nlp(model_name)
    resolved = []
    prev_subject = None
    prev_noun_phrases = []

    for sent_text in sentences:
        doc = nlp(sent_text)

        # Extract noun phrases and subjects from this sentence for future reference
        current_subjects = []
        current_nps = []
        for chunk in doc.noun_chunks:
            current_nps.append(chunk.text)
            if chunk.root.dep_ in ("nsubj", "nsubjpass"):
                current_subjects.append(chunk.text)

        # Attempt to resolve pronouns using previous sentence context
        resolved_text = sent_text
        if prev_subject or prev_noun_phrases:
            for token in doc:
                if token.text.lower() in PRONOUNS_TO_RESOLVE:
                    # Skip idiomatic expressions like "it is clear", "it can be seen"
                    if token.text.lower() == "it":
                        # Check if this is an impersonal 'it' construction
                        next_token = doc[token.i + 1] if token.i + 1 < len(doc) else None
                        if next_token:
                            bigram = f"{token.text.lower()} {next_token.text.lower()}"
                            if bigram in IDIOMATIC_IT_PATTERNS:
                                continue

                    referent = _find_best_referent(
                        token.text.lower(), prev_subject, prev_noun_phrases
                    )
                    if referent:
                        # Replace the pronoun with the referent
                        # Use word boundary-aware replacement
                        pattern = r"\b" + re.escape(token.text) + r"\b"
                        resolved_text = re.sub(
                            pattern, referent, resolved_text, count=1
                        )

        resolved.append(resolved_text)

        # Update context for next sentence
        if current_subjects:
            prev_subject = current_subjects[0]
        prev_noun_phrases = current_nps if current_nps else prev_noun_phrases

    return resolved


def _find_best_referent(
    pronoun: str,
    prev_subject: str | None,
    prev_noun_phrases: list[str],
) -> str | None:
    """
    Find the best referent for a pronoun from previous sentence context.
    """
    if not prev_subject and not prev_noun_phrases:
        return None

    # "it" / "its" → singular subject
    if pronoun in ("it", "its"):
        return prev_subject

    # "they" / "their" / "them" → plural or first NP
    if pronoun in ("they", "their", "them"):
        return prev_subject

    # "this" / "these" → first noun phrase from previous sentence
    if pronoun in ("this", "these"):
        if prev_noun_phrases:
            return prev_noun_phrases[0]
        return prev_subject

    return None


# ──────────────────────────────────────────────
# Word Count
# ──────────────────────────────────────────────

def count_words(text: str) -> int:
    """Count words in the text (simple whitespace-based)."""
    words = text.split()
    return len(words)


# ──────────────────────────────────────────────
# Lexical Metrics
# ──────────────────────────────────────────────

# Academic Word List (AWL) — Sublist 1 most frequent academic words
# Source: Averil Coxhead's Academic Word List
AWL_WORDS = {
    "analysis", "approach", "area", "assessment", "assume", "authority",
    "available", "benefit", "concept", "consistent", "constitutional",
    "context", "contract", "create", "data", "definition", "derived",
    "distribution", "economic", "environment", "established", "estimate",
    "evidence", "export", "factor", "financial", "formula", "function",
    "identified", "income", "indicate", "individual", "interpretation",
    "involved", "issues", "labour", "legal", "legislation", "major",
    "method", "occur", "percent", "period", "policy", "principle",
    "procedure", "process", "required", "research", "response", "role",
    "section", "sector", "significant", "similar", "source", "specific",
    "structure", "theory", "variable",
    # Extended with common IELTS Task 1 academic vocabulary
    "proportion", "percentage", "approximately", "considerably",
    "dramatically", "gradually", "significantly", "steadily",
    "substantially", "respectively", "whereas", "overall",
    "subsequently", "furthermore", "moreover", "consequently",
    "illustrated", "depicted", "represented", "comprised",
    "constituted", "accounted", "compared", "contrast",
    "fluctuated", "stabilized", "peaked", "declined",
}


def compute_lexical_metrics(
    text: str, model_name: str = "en_core_web_sm"
) -> dict:
    """
    Compute lexical metrics for Lexical Resource scoring.

    Returns:
        {
            "total_words": int,
            "unique_words": int,
            "ttr": float,                    # Type-Token Ratio
            "academic_word_count": int,
            "academic_word_density": float,
            "spelling_errors": int,          # placeholder — filled by LanguageTool
            "trend_word_repetition": int,
            "unique_trend_phrases": int,
        }
    """
    nlp = get_nlp(model_name)
    doc = nlp(text.lower())

    # Filter to meaningful tokens (no punct, no space, no stop words for TTR)
    content_tokens = [
        token.lemma_ for token in doc
        if not token.is_punct and not token.is_space
    ]
    all_tokens = [
        token.text for token in doc
        if not token.is_punct and not token.is_space
    ]

    total = len(all_tokens)
    unique = len(set(content_tokens))

    # Academic Word Density
    academic_count = sum(1 for t in content_tokens if t in AWL_WORDS)

    # Trend word repetition detection
    trend_words = _count_trend_word_repetitions(text.lower())

    return {
        "total_words": total,
        "unique_words": unique,
        "ttr": round(unique / total, 3) if total else 0.0,
        "academic_word_count": academic_count,
        "academic_word_density": round(academic_count / total, 3) if total else 0.0,
        "spelling_errors": 0,  # Will be updated by LanguageTool in Agent 3
        "trend_word_repetition": trend_words["total_repetitions"],
        "unique_trend_phrases": trend_words["unique_phrases"],
    }


def _count_trend_word_repetitions(text: str) -> dict:
    """
    Count repetitions of trend-describing words/phrases.
    IELTS penalizes overuse of the same trend words (e.g., always using
    'increase' instead of varying with 'rise', 'grow', 'climb', etc.)
    """
    # Major trend word groups
    trend_groups = {
        "increase": ["increase", "increased", "increasing", "increases"],
        "decrease": ["decrease", "decreased", "decreasing", "decreases"],
        "rise": ["rise", "rose", "risen", "rising", "rises"],
        "fall": ["fall", "fell", "fallen", "falling", "falls"],
        "grow": ["grow", "grew", "grown", "growing", "grows"],
        "drop": ["drop", "dropped", "dropping", "drops"],
        "decline": ["decline", "declined", "declining", "declines"],
        "remain": ["remain", "remained", "remaining", "remains"],
        "fluctuate": ["fluctuate", "fluctuated", "fluctuating"],
        "peak": ["peak", "peaked", "peaking"],
        "climb": ["climb", "climbed", "climbing"],
        "surge": ["surge", "surged", "surging"],
        "plummet": ["plummet", "plummeted", "plummeting"],
        "soar": ["soar", "soared", "soaring"],
        "dip": ["dip", "dipped", "dipping"],
        "level off": ["level off", "leveled off", "levelled off"],
    }

    words = text.split()
    text_lower = text.lower()

    # Count how many times each trend group appears
    group_counts = {}
    used_groups = set()
    for group_name, variants in trend_groups.items():
        count = 0
        for variant in variants:
            count += text_lower.count(variant)
        if count > 0:
            group_counts[group_name] = count
            used_groups.add(group_name)

    # Max repetition = the highest count of any single trend group
    # This measures how much the candidate overuses one specific word family
    # e.g., using "increase/increased/increasing" 5 times → repetition = 5
    total_reps = max(group_counts.values()) if group_counts else 0

    return {
        "total_repetitions": total_reps,
        "unique_phrases": len(used_groups),
        "group_counts": group_counts,
    }
