"""
Rubric Loader — Reads and parses IELTS Band Descriptor JSON files.
Provides utility functions to load rubric data and match scores to band levels.
"""

import json
from pathlib import Path
from functools import lru_cache

# Path to the rubrics directory (relative to project root)
RUBRICS_DIR = Path(__file__).resolve().parents[3] / "rubrics"


@lru_cache(maxsize=8)
def load_rubric(criterion: str) -> dict:
    """
    Load a rubric JSON file by criterion name.

    Args:
        criterion: One of 'task_achievement', 'coherence_cohesion',
                   'lexical_resource', 'grammar_accuracy'

    Returns:
        Parsed JSON dictionary with band levels as keys.
    """
    filepath = RUBRICS_DIR / f"{criterion}.json"
    if not filepath.exists():
        raise FileNotFoundError(f"Rubric file not found: {filepath}")

    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def get_band_description(criterion: str, band: int) -> str:
    """
    Get the official text description for a specific band level.

    Args:
        criterion: Rubric criterion name
        band: Band level (0-9)

    Returns:
        Description string for the given band level.
    """
    rubric = load_rubric(criterion)
    band_key = str(band)
    if band_key not in rubric:
        return ""
    return rubric[band_key].get("description", "")


def get_band_heuristics(criterion: str, band: int) -> dict:
    """
    Get the quantitative heuristics for a specific band level.

    Args:
        criterion: Rubric criterion name
        band: Band level (0-9)

    Returns:
        Heuristics dictionary for the given band level.
    """
    rubric = load_rubric(criterion)
    band_key = str(band)
    if band_key not in rubric:
        return {}
    return rubric[band_key].get("heuristics", {})


def get_all_descriptions(criterion: str) -> str:
    """
    Get all band descriptions formatted as a string for LLM prompt injection.

    Args:
        criterion: Rubric criterion name

    Returns:
        Formatted string with all band levels and descriptions.
    """
    rubric = load_rubric(criterion)
    lines = []
    for band in range(9, -1, -1):
        band_key = str(band)
        if band_key in rubric and "description" in rubric[band_key]:
            desc = rubric[band_key]["description"]
            lines.append(f"Band {band}: {desc}")
    return "\n".join(lines)


# criterion -> [(metric key, heuristic key, kind)].  "min": metric must reach
# the threshold; "max": metric must not exceed it; "flag": a feature the band
# requires to be present.
_HEURISTIC_SPECS: dict[str, list[tuple[str, str, str]]] = {
    "task_achievement": [
        ("coverage_rate", "min_coverage_rate", "min"),
        ("contradiction_rate", "max_contradiction_rate", "max"),
        ("has_overview", "requires_overview", "flag"),
    ],
    "grammar_accuracy": [
        ("complex_sentence_ratio", "min_complex_sentence_ratio", "min"),
        ("grammar_errors_per_100_words", "max_grammar_errors_per_100_words", "max"),
        ("sentence_variety_score", "min_sentence_variety_score", "min"),
    ],
    "lexical_resource": [
        ("ttr", "min_ttr", "min"),
        ("academic_word_density", "min_academic_word_density", "min"),
        ("spelling_errors", "max_spelling_errors", "max"),
        ("trend_word_repetition", "max_trend_word_repetition", "max"),
    ],
    "coherence_cohesion": [
        ("cohesive_device_count", "min_cohesive_devices", "min"),
        ("repeated_connectors", "max_repeated_connectors", "max"),
        ("paragraph_count", "min_paragraph_count", "min"),
    ],
}


def match_score_to_band(criterion: str, metrics: dict) -> float:
    """
    Match computed metrics against rubric heuristics to determine band score.
    Uses a top-down approach: starts from Band 9 and finds the highest band
    whose heuristics the metrics satisfy.  A half band is awarded when the
    metrics are also at least halfway to every threshold of the next band.

    Args:
        criterion: Rubric criterion name
        metrics: Dictionary of computed metric values

    Returns:
        Band score as a float (e.g., 7.0, 6.5)
    """
    rubric = load_rubric(criterion)

    for band in range(9, 0, -1):
        heuristics = _band_heuristics(rubric, band)
        if not heuristics:
            continue
        if _metrics_satisfy_heuristics(criterion, metrics, heuristics):
            next_heuristics = _band_heuristics(rubric, band + 1)
            if next_heuristics and _is_halfway_to_next_band(
                criterion, metrics, heuristics, next_heuristics
            ):
                return band + 0.5
            return float(band)

    return 1.0  # Fallback to Band 1


def _band_heuristics(rubric: dict, band: int) -> dict:
    return rubric.get(str(band), {}).get("heuristics", {})


def _metric_value(metrics: dict, metric_key: str, kind: str) -> float:
    """Read a metric, treating a missing one as the worst possible value."""
    if kind == "flag":
        return bool(metrics.get(metric_key, False))
    return metrics.get(metric_key, 0 if kind == "min" else float("inf"))


def _metrics_satisfy_heuristics(
    criterion: str, metrics: dict, heuristics: dict
) -> bool:
    """
    Check if computed metrics satisfy the heuristics for a band level.
    Each criterion has different heuristic keys.
    """
    for metric_key, heuristic_key, kind in _HEURISTIC_SPECS.get(criterion, []):
        if heuristic_key not in heuristics:
            continue
        threshold = heuristics[heuristic_key]
        value = _metric_value(metrics, metric_key, kind)
        if kind == "min" and value < threshold:
            return False
        if kind == "max" and value > threshold:
            return False
        if kind == "flag" and threshold and not value:
            return False
    return True


def _is_halfway_to_next_band(
    criterion: str, metrics: dict, heuristics: dict, next_heuristics: dict
) -> bool:
    """
    Return whether metrics that satisfy one band sit in the upper half of the
    gap to the next band on every condition they still miss.  A missing
    required feature (e.g. no overview) can never earn the half band.
    """
    for metric_key, heuristic_key, kind in _HEURISTIC_SPECS.get(criterion, []):
        if heuristic_key not in next_heuristics:
            continue
        target = next_heuristics[heuristic_key]
        value = _metric_value(metrics, metric_key, kind)

        if kind == "flag":
            if target and not value:
                return False
            continue

        current = heuristics.get(heuristic_key, target)
        midpoint = (current + target) / 2
        if kind == "min" and value < target and value < midpoint:
            return False
        if kind == "max" and value > target and value > midpoint:
            return False
    return True
