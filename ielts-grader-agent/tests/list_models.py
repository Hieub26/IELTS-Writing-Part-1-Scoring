"""List models available for the Gemini API key."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from google import genai
from app.config import settings

def list_gemini_models():
    api_key = settings.gemini_api_key
    if not api_key:
        print("[ERROR] GEMINI_API_KEY is not configured in .env")
        return
        
    print("Connecting to Gemini API to query available models...")
    try:
        client = genai.Client(api_key=api_key)
        
        # Get list of models
        models = client.models.list()
        
        print("\nAvailable Models for your API Key:")
        print("=" * 60)
        
        gemini_models = []
        for model in models:
            # We want to display Gemini models
            name = model.name
            display_name = model.display_name or "No display name"
            supported_actions = model.supported_actions or []
            
            # Filter to show main text/image generation models (Gemini models)
            if "gemini" in name.lower() and "generateContent" in supported_actions:
                gemini_models.append((name, display_name))
                
        # Sort and print
        gemini_models.sort()
        for idx, (name, display) in enumerate(gemini_models):
            print(f"[{idx+1}] {name} ({display})")
            
        print("=" * 60)
        print(f"Total available generation models: {len(gemini_models)}")
        
    except Exception as e:
        print(f"[ERROR] Failed to fetch models list.")
        print(f"Details: {str(e)}")

if __name__ == "__main__":
    list_gemini_models()
