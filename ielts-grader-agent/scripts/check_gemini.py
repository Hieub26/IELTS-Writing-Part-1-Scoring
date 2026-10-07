"""Check that the configured Gemini API key and models work.

Usage: python scripts/check_gemini.py

This makes a real, billable API request, so it is a manual script and is
deliberately not part of the pytest suite.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from google import genai
from app.config import settings


def main() -> int:
    api_key = settings.gemini_api_key
    if not api_key:
        print("[ERROR] GEMINI_API_KEY is not set in your .env file.")
        return 1

    print(f"VLM Model: {settings.gemini_model_vlm}")
    print(f"LLM Model: {settings.gemini_model_llm}")

    try:
        client = genai.Client(api_key=api_key)
        print("Sending test request to Gemini API...")
        response = client.models.generate_content(
            model=settings.gemini_model_llm,
            contents="Say 'API is active and working!' in exactly one line."
        )
    except Exception as e:
        print("[ERROR] Gemini API call failed.")
        print(f"Details: {str(e)}")
        return 1

    print(f"[OK] Success! Response from Gemini: {(response.text or '').strip()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
