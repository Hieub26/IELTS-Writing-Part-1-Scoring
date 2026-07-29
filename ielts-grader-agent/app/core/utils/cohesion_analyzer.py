"""
Cohesion Analyzer — Quantitative analysis of cohesive devices and referencing.
Used by the Coherence & Cohesion Agent (Agent 4).

Provides:
  - Cohesive device detection and categorization
  - Repetition analysis for connectors
  - Paragraph structure analysis
  - Referencing quality assessment
"""

from __future__ import annotations

import re
from collections import Counter
from functools import lru_cache

from app.core.utils.rubric_loader import load_rubric


@lru_cache(maxsize=1)
def get_cohesive_devices_dict() -> dict[str, list[str]]:
    """Load the cohesive devices dictionary from the CC rubric."""
    rubric = load_rubric("coherence_cohesion")
    return rubric.get("cohesive_devices_dictionary", {})


def analyze_cohesion(text: str) -> dict:
    """
    Analyze the cohesive devices usage in the essay.

    Returns:
        {
            "cohesive_device_count": int,          # Total unique devices found
            "devices_by_category": dict,            # Category → list of found devices
            "category_coverage": dict,              # Category → count
            "repeated_connectors": int,             # Devices used 3+ times
            "overused_devices": list[str],          # Specific devices overused
            "device_variety_score": float,          # 0-1, higher = more varied
            "total_device_occurrences": int,
        }
    """
    devices_dict = get_cohesive_devices_dict()
    text_lower = text.lower()

    found_devices = {}         # device → count
    devices_by_category = {}   # category → [devices found]
    category_coverage = {}     # category → total count

    for category, device_list in devices_dict.items():
        found_in_category = []
        cat_count = 0

        for device in device_list:
            # Use word boundary matching to avoid partial matches
            pattern = r"\b" + re.escape(device) + r"\b"
            matches = re.findall(pattern, text_lower)
            count = len(matches)
            if count > 0:
                found_devices[device] = count
                found_in_category.append(device)
                cat_count += count

        if found_in_category:
            devices_by_category[category] = found_in_category
            category_coverage[category] = cat_count

    # Count repeated connectors (any device appearing 3+ times)
    overused = [d for d, c in found_devices.items() if c >= 3]
    repeated_count = len(overused)

    # Device variety: how many categories are covered
    total_categories = len(devices_dict)
    covered_categories = len(devices_by_category)
    variety_score = covered_categories / total_categories if total_categories else 0

    total_occurrences = sum(found_devices.values())

    return {
        "cohesive_device_count": len(found_devices),
        "devices_by_category": devices_by_category,
        "category_coverage": category_coverage,
        "repeated_connectors": repeated_count,
        "overused_devices": overused,
        "device_variety_score": round(variety_score, 3),
        "total_device_occurrences": total_occurrences,
    }


def analyze_paragraph_structure(text: str) -> dict:
    """
    Analyze the paragraph structure of the essay.
    IELTS Task 1 should ideally have:
      - Introduction (paraphrase of the question)
      - Overview (summary of main trends)
      - 1-2 Detail paragraphs

    Returns:
        {
            "paragraph_count": int,
            "paragraph_lengths": list[int],      # word count per paragraph
            "has_intro": bool,
            "has_overview": bool,
            "structure_score": float,             # 0-1
        }
    """
    # Split into paragraphs (by double newline or significant whitespace)
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n|\n{2,}", text) if p.strip()]

    # If only single newlines, try splitting by single newline
    if len(paragraphs) <= 1:
        paragraphs = [p.strip() for p in text.split("\n") if p.strip()]

    paragraph_lengths = [len(p.split()) for p in paragraphs]
    para_count = len(paragraphs)

    # Detect overview paragraph: contains overview/summary keywords
    overview_keywords = [
        "overall", "in general", "it is clear", "it can be seen",
        "to summarize", "to summarise", "in summary", "generally",
        "on the whole", "it is evident", "it is noticeable",
        "the most striking", "the most notable", "a glance",
    ]

    has_overview = False
    for para in paragraphs:
        para_lower = para.lower()
        if any(kw in para_lower for kw in overview_keywords):
            has_overview = True
            break

    # Detect introduction: first paragraph often contains 'shows', 'illustrates',
    # 'depicts', 'presents', 'gives information', 'compares'
    intro_keywords = [
        "shows", "illustrates", "depicts", "presents",
        "gives information", "compares", "provided", "chart",
        "graph", "table", "diagram", "figure", "pie",
    ]
    has_intro = False
    if paragraphs:
        first_para = paragraphs[0].lower()
        if any(kw in first_para for kw in intro_keywords):
            has_intro = True

    # Structure scoring
    structure_score = 0.0
    if has_intro:
        structure_score += 0.3
    if has_overview:
        structure_score += 0.3
    if para_count >= 3:
        structure_score += 0.2
    elif para_count >= 2:
        structure_score += 0.1
    # Bonus for balanced paragraph lengths
    if para_count >= 2:
        lengths = paragraph_lengths[1:]  # Exclude intro
        if lengths:
            avg = sum(lengths) / len(lengths)
            if avg > 0:
                deviations = [abs(l - avg) / avg for l in lengths]
                balance = 1.0 - min(sum(deviations) / len(deviations), 1.0)
                structure_score += 0.2 * balance

    return {
        "paragraph_count": para_count,
        "paragraph_lengths": paragraph_lengths,
        "has_intro": has_intro,
        "has_overview": has_overview,
        "structure_score": round(min(structure_score, 1.0), 3),
    }


def analyze_referencing(
    sentences: list[str],
    nlp_model_name: str = "en_core_web_sm",
) -> dict:
    """
    Analyze referencing quality using spaCy.
    Checks if pronouns (it, they, this, these) are used clearly
    or if they create ambiguity.

    Returns:
        {
            "total_pronouns": int,
            "ambiguous_references": int,
            "clear_references": int,
            "referencing_score": float,    # 0-1, higher = clearer referencing
        }
    """
    from app.core.utils.linguistics import get_nlp

    nlp = get_nlp(nlp_model_name)
    pronouns_to_check = {"it", "they", "this", "these", "its", "their"}

    total_pronouns = 0
    ambiguous = 0

    for i, sent_text in enumerate(sentences):
        doc = nlp(sent_text)
        for token in doc:
            if token.text.lower() in pronouns_to_check:
                total_pronouns += 1
                # A pronoun at the start of an essay (first sentence) is ambiguous
                if i == 0:
                    ambiguous += 1
                    continue
                # Check if there's a clear antecedent in the same sentence
                has_antecedent = any(
                    chunk.root.dep_ in ("nsubj", "nsubjpass", "dobj", "pobj")
                    and chunk.root.i < token.i
                    for chunk in doc.noun_chunks
                )
                if not has_antecedent:
                    # Not necessarily ambiguous — could reference previous sentence
                    # We'll be lenient here
                    pass

    clear = total_pronouns - ambiguous
    ref_score = clear / total_pronouns if total_pronouns > 0 else 1.0

    return {
        "total_pronouns": total_pronouns,
        "ambiguous_references": ambiguous,
        "clear_references": clear,
        "referencing_score": round(ref_score, 3),
    }
