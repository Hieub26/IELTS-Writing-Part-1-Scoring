"""Test Gemini API key validity."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from google import genai
from app.config import settings

def test_gemini_api():
    print("Loading settings...")
    api_key = settings.gemini_api_key
    if not api_key:
        print("❌ Error: GEMINI_API_KEY is not set in your .env file.")
        return
        
    masked_key = api_key[:6] + "..." + api_key[-4:] if len(api_key) > 10 else "too short"
    print(f"Checking API Key: {masked_key}")
    print(f"VLM Model: {settings.gemini_model_vlm}")
    print(f"LLM Model: {settings.gemini_model_llm}")
    
    try:
        client = genai.Client(api_key=api_key)
        print("Sending test request to Gemini API...")
        response = client.models.generate_content(
            model=settings.gemini_model_llm,
            contents="Say 'API is active and working!' in exactly one line."
        )
        print(f"[OK] Success! Response from Gemini: {response.text.strip()}")
    except Exception as e:
        print(f"[ERROR] Gemini API call failed.")
        print(f"Details: {str(e)}")

if __name__ == "__main__":
    test_gemini_api()
