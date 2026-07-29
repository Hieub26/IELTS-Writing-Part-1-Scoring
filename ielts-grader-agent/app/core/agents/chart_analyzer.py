"""Gemini Vision chart analyzer for the IELTS grading pipeline."""

from __future__ import annotations

import json
import re
import time
from pathlib import Path

from google import genai
from google.genai import types

from app.config import settings
from app.core.state import GraderState
from app.core.utils.prompt_templates import CHART_ANALYZER_PROMPT


MAX_VLM_RETRIES = 3


def chart_analyzer_node(state: GraderState) -> dict:
    """Extract chart data with Gemini Vision and return strict JSON."""
    image_path = state.get("image_path", "")
    path = Path(image_path)
    if not image_path or not path.exists():
        return _error_result(f"Image file not found: {image_path}")
    if not settings.gemini_api_key:
        return _error_result("GEMINI_API_KEY is not configured for chart analysis.")

    last_error = ""
    for attempt in range(MAX_VLM_RETRIES):
        try:
            chart_data = _analyze_with_gemini(path)
            if not chart_data or not chart_data.get("key_trends"):
                raise ValueError("Gemini returned empty chart data or missing key trends")
            chart_type = chart_data.get("chart_type", "unknown").lower()
            valid_types = {"bar", "line", "pie", "table", "mixed", "map", "process"}
            return {
                "chart_type": chart_type if chart_type in valid_types else "mixed",
                "chart_data": chart_data,
                "chart_analysis_error": "",
            }
        except Exception as exc:
            last_error = f"Gemini Vision error (attempt {attempt + 1}): {exc}"
            if not _is_transient_error(exc) or attempt == MAX_VLM_RETRIES - 1:
                break
            time.sleep(2 ** attempt)

    return _error_result(last_error)


def _analyze_with_gemini(image_path: Path) -> dict:
    """Run one Gemini multimodal request and parse its JSON response."""
    mime_type = _image_mime_type(image_path)
    client = genai.Client(api_key=settings.gemini_api_key)
    image_part = types.Part.from_bytes(data=image_path.read_bytes(), mime_type=mime_type)
    response = client.models.generate_content(
        model=settings.gemini_model_vlm,
        contents=[image_part, CHART_ANALYZER_PROMPT],
        config=types.GenerateContentConfig(
            temperature=0,
            max_output_tokens=4096,
            response_mime_type="application/json",
        ),
    )
    if not response.text:
        raise ValueError("Gemini returned an empty response")
    return _parse_vlm_json(response.text)


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


def _is_transient_error(error: Exception) -> bool:
    message = str(error)
    return "429" in message or "500" in message or "503" in message


def _error_result(error: str) -> dict:
    return {"chart_type": "unknown", "chart_data": {}, "chart_analysis_error": error}


def _parse_vlm_json(raw_text: str) -> dict:
    """Parse JSON, tolerating Markdown code fences around the model response."""
    text = raw_text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*\n?", "", text)
        text = re.sub(r"\n?```\s*$", "", text)
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        text = text[start:end + 1]
    return json.loads(text)


def route_chart_type(state: GraderState) -> str:
    """Route chart types for callers that use this legacy helper."""
    if state.get("chart_analysis_error"):
        return "error"
    return "structured" if state.get("chart_type") in ("bar", "line") else "complex"
