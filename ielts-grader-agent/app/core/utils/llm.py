"""
Gemini helpers shared by every LLM-backed agent.

Keeps one client, one retry policy and one JSON parser so the agents cannot
drift apart in how they detect transient failures or read model output.
"""

from __future__ import annotations

import json
import re
import time
from functools import lru_cache

from google import genai
from google.genai import types

from app.config import settings


MAX_RETRIES = 3

# Gemini 2.5 models spend "thinking" tokens out of the same budget as the
# visible answer.  A tight limit truncates the JSON mid-object, so every call
# gets generous headroom by default.
DEFAULT_MAX_OUTPUT_TOKENS = 8192

_TRANSIENT_CODES = {429, 500, 503, 504}
_TRANSIENT_MARKERS = ("429", "500", "503", "504", "UNAVAILABLE", "RESOURCE_EXHAUSTED")


def llm_available() -> bool:
    """Return whether a Gemini API key is configured."""
    return bool(settings.gemini_api_key)


@lru_cache(maxsize=2)
def _client_for(api_key: str) -> genai.Client:
    return genai.Client(api_key=api_key)


def get_client() -> genai.Client:
    """Return the cached Gemini client for the configured API key."""
    if not settings.gemini_api_key:
        raise RuntimeError("GEMINI_API_KEY is not configured.")
    return _client_for(settings.gemini_api_key)


def is_transient_error(error: Exception) -> bool:
    """Return whether an API failure is worth retrying."""
    if getattr(error, "code", None) in _TRANSIENT_CODES:
        return True
    message = str(error).upper()
    return any(marker in message for marker in _TRANSIENT_MARKERS)


def parse_json_object(raw_text: str) -> dict:
    """Parse a JSON object, tolerating Markdown fences and surrounding prose.

    Raises:
        ValueError: If the text does not contain a valid JSON object.
    """
    text = (raw_text or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*\n?", "", text)
        text = re.sub(r"\n?```\s*$", "", text)

    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        raise ValueError("Model response does not contain a JSON object")

    result = json.loads(text[start : end + 1])
    if not isinstance(result, dict):
        raise ValueError("Model response is not a JSON object")
    return result


def generate_json(
    contents,
    *,
    model: str | None = None,
    temperature: float = 0.2,
    max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
) -> dict:
    """Call Gemini in JSON mode and return the parsed object.

    Transient capacity/rate-limit failures are retried with exponential
    backoff.  Any other failure, an empty response, or malformed JSON raises
    so the caller can choose an explicit fallback.
    """
    client = get_client()
    config = types.GenerateContentConfig(
        temperature=temperature,
        max_output_tokens=max_output_tokens,
        response_mime_type="application/json",
    )

    for attempt in range(MAX_RETRIES):
        try:
            response = client.models.generate_content(
                model=model or settings.gemini_model_llm,
                contents=contents,
                config=config,
            )
        except Exception as exc:
            if not is_transient_error(exc) or attempt == MAX_RETRIES - 1:
                raise
            time.sleep(2 ** attempt)
            continue

        if not response.text:
            raise ValueError("Gemini returned an empty response")
        return parse_json_object(response.text)

    raise RuntimeError("Gemini request failed after retries")
