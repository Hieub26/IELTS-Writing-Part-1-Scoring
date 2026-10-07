"""Gemini Vision chart analyzer for the IELTS grading pipeline."""

from __future__ import annotations

import copy
import hashlib
from pathlib import Path

from google.genai import types

from app.config import settings
from app.core.state import GraderState
from app.core.utils.llm import generate_json, llm_available
from app.core.utils.prompt_templates import CHART_ANALYZER_PROMPT


VALID_CHART_TYPES = {"bar", "line", "pie", "table", "mixed", "map", "process"}

# Successful extractions keyed by image content.  The same chart must yield the
# same key trends on every run, otherwise TA coverage (and therefore the band)
# changes between two gradings of an identical submission.
_CHART_CACHE: dict[str, dict] = {}
_CHART_CACHE_MAX_ENTRIES = 32


def chart_analyzer_node(state: GraderState) -> dict:
    """Extract chart data with Gemini Vision and return strict JSON."""
    image_path = state.get("image_path", "")
    path = Path(image_path)
    if not image_path or not path.exists():
        return _error_result(f"Image file not found: {image_path}")
    if not llm_available():
        return _error_result("GEMINI_API_KEY is not configured for chart analysis.")

    task_prompt = (state.get("task_prompt") or "").strip()
    image_bytes = path.read_bytes()
    cache_key = _cache_key(image_bytes, task_prompt)

    chart_data = _CHART_CACHE.get(cache_key)
    if chart_data is None:
        try:
            chart_data = _analyze_with_gemini(
                image_bytes, _image_mime_type(path), task_prompt
            )
            if not chart_data.get("key_trends"):
                raise ValueError("Gemini returned empty chart data or missing key trends")
        except Exception as exc:
            return _error_result(f"Gemini Vision error: {exc}")
        _remember(cache_key, chart_data)

    chart_data = copy.deepcopy(chart_data)
    chart_type = str(chart_data.get("chart_type", "unknown")).lower()
    return {
        "chart_type": chart_type if chart_type in VALID_CHART_TYPES else "mixed",
        "chart_data": chart_data,
        "chart_analysis_error": "",
    }


def _analyze_with_gemini(image_bytes: bytes, mime_type: str, task_prompt: str) -> dict:
    """Run one Gemini multimodal request and parse its JSON response."""
    prompt = CHART_ANALYZER_PROMPT
    if task_prompt:
        prompt += (
            "\n## Task statement shown to the candidate (context for the image):\n"
            f"{task_prompt}\n"
        )
    image_part = types.Part.from_bytes(data=image_bytes, mime_type=mime_type)
    return generate_json(
        [image_part, prompt],
        model=settings.gemini_model_vlm,
        temperature=0,
    )


def _cache_key(image_bytes: bytes, task_prompt: str) -> str:
    digest = hashlib.sha256(image_bytes)
    digest.update(settings.gemini_model_vlm.encode("utf-8"))
    digest.update(task_prompt.encode("utf-8"))
    return digest.hexdigest()


def _remember(cache_key: str, chart_data: dict) -> None:
    if len(_CHART_CACHE) >= _CHART_CACHE_MAX_ENTRIES:
        _CHART_CACHE.pop(next(iter(_CHART_CACHE)))
    _CHART_CACHE[cache_key] = chart_data


def _image_mime_type(image_path: Path) -> str:
    """Return a Gemini-supported MIME type for an uploaded chart image."""
    mime_types = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".webp": "image/webp",
        ".gif": "image/gif",
    }
    return mime_types.get(image_path.suffix.lower(), "image/png")


def _error_result(error: str) -> dict:
    return {"chart_type": "unknown", "chart_data": {}, "chart_analysis_error": error}
