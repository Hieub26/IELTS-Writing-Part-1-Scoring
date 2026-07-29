"""
IELTS Multi-Agent Grading System — Configuration.
Uses pydantic-settings to load environment variables with validation and defaults.
"""

from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # --- Google Gemini API (text synthesis only) ---
    gemini_api_key: str = Field(
        default="",
        description="Google Gemini API key",
    )
    gemini_model_vlm: str = Field(
        default="gemini-2.5-flash",
        description="Gemini model for chart image analysis",
    )
    gemini_model_llm: str = Field(
        default="gemini-2.5-flash",
        description="Gemini model for LLM (Chief Examiner, CC Agent)",
    )

    # --- NLP Models ---
    nli_model_name: str = Field(
        default="cross-encoder/nli-deberta-v3-base",
        description="Cross-Encoder model for NLI (Feature Grounding)",
    )
    spacy_model: str = Field(
        default="en_core_web_sm",
        description="spaCy language model",
    )

    # --- Self-Correction Loop ---
    max_correction_loops: int = Field(
        default=2,
        description="Maximum number of self-correction iterations",
    )

    # --- Confidence Thresholds ---
    min_confidence_threshold: float = Field(
        default=0.5,
        description="Minimum confidence score to accept agent output",
    )

    # --- IELTS Constraints ---
    min_word_count: int = Field(
        default=150,
        description="Minimum word count for IELTS Task 1 (penalty below this)",
    )

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": False,
        "extra": "ignore",
    }

# Singleton instance
settings = Settings()
