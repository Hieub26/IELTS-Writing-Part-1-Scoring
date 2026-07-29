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


def match_score_to_band(criterion: str, metrics: dict) -> float:
    """
    Match computed metrics against rubric heuristics to determine band score.
    Uses a top-down approach: starts from Band 9 and finds the highest band
    whose heuristics the metrics satisfy.

    Args:
        criterion: Rubric criterion name
        metrics: Dictionary of computed metric values

    Returns:
        Band score as a float (e.g., 7.0, 6.5)
    """
    rubric = load_rubric(criterion)

    for band in range(9, 0, -1):
        band_key = str(band)
        if band_key not in rubric:
            continue
        heuristics = rubric[band_key].get("heuristics", {})
        if not heuristics:
            continue
        if _metrics_satisfy_heuristics(criterion, metrics, heuristics):
            return float(band)

    return 1.0  # Fallback to Band 1


def _metrics_satisfy_heuristics(
    criterion: str, metrics: dict, heuristics: dict
) -> bool:
    """
    Check if computed metrics satisfy the heuristics for a band level.
    Each criterion has different heuristic keys.
    """
    if criterion == "task_achievement":
        coverage = metrics.get("coverage_rate", 0)
        contradiction = metrics.get("contradiction_rate", 1)
        has_overview = metrics.get("has_overview", False)
        min_cov = heuristics.get("min_coverage_rate", 0)
        max_con = heuristics.get("max_contradiction_rate", 1)
        req_overview = heuristics.get("requires_overview", False)

        if coverage < min_cov:
            return False
        if contradiction > max_con:
            return False
        if req_overview and not has_overview:
            return False
        return True

    elif criterion == "grammar_accuracy":
        csr = metrics.get("complex_sentence_ratio", 0)
        errors_per_100 = metrics.get("grammar_errors_per_100_words", 100)
        variety = metrics.get("sentence_variety_score", 0)

        if csr < heuristics.get("min_complex_sentence_ratio", 0):
            return False
        if errors_per_100 > heuristics.get("max_grammar_errors_per_100_words", 100):
            return False
        if variety < heuristics.get("min_sentence_variety_score", 0):
            return False
        return True

    elif criterion == "lexical_resource":
        ttr = metrics.get("ttr", 0)
        awd = metrics.get("academic_word_density", 0)
        spelling = metrics.get("spelling_errors", 100)
        trend_rep = metrics.get("trend_word_repetition", 100)

        if ttr < heuristics.get("min_ttr", 0):
            return False
        if awd < heuristics.get("min_academic_word_density", 0):
            return False
        if spelling > heuristics.get("max_spelling_errors", 100):
            return False
        if trend_rep > heuristics.get("max_trend_word_repetition", 100):
            return False
        return True

    elif criterion == "coherence_cohesion":
        devices = metrics.get("cohesive_device_count", 0)
        repeated = metrics.get("repeated_connectors", 100)
        para_count = metrics.get("paragraph_count", 0)

        if devices < heuristics.get("min_cohesive_devices", 0):
            return False
        if repeated > heuristics.get("max_repeated_connectors", 100):
            return False
        if para_count < heuristics.get("min_paragraph_count", 0):
            return False
        return True

    # Unknown criterion — accept by default
    return True
