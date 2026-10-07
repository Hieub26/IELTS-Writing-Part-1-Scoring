"""
LangGraph State Definition — GraderState.
This TypedDict defines the shared memory for all agents in the grading pipeline.
"""

from __future__ import annotations
from typing import TypedDict


class GraderState(TypedDict):
    """Shared state for the IELTS Multi-Agent Grading pipeline."""

    # ═══════════════════════════════════════════
    # Inputs
    # ═══════════════════════════════════════════
    image_path: str                     # Path to the uploaded chart image
    essay_text: str                     # Candidate's essay text
    task_prompt: str                    # Optional task statement shown with the chart
    word_count: int                     # Word count (computed at start)

    # ═══════════════════════════════════════════
    # Agent 1: Chart Analyzer Output
    # ═══════════════════════════════════════════
    chart_type: str                     # "bar"|"line"|"pie"|"table"|"mixed"|"map"|"process"
    chart_data: dict                    # Full JSON extracted from chart
    chart_analysis_error: str           # Error message if VLM fails

    # ═══════════════════════════════════════════
    # Agent 2: Feature Grounding (TA) Output
    # ═══════════════════════════════════════════
    ta_score: float                     # Task Achievement score (0-9)
    ta_confidence: float                # Confidence of TA assessment
    ta_details: dict                    # Coverage rate, contradiction rate, etc.
    grounding_report: list[dict]        # Per-sentence NLI analysis

    # ═══════════════════════════════════════════
    # Agent 3: Grammar & Lexical Output
    # ═══════════════════════════════════════════
    gra_score: float                    # Grammatical Range & Accuracy (0-9)
    gra_confidence: float               # Confidence of GRA assessment
    lr_score: float                     # Lexical Resource (0-9)
    lr_confidence: float                # Confidence of LR assessment
    grammar_errors: list[dict]          # List of errors with positions
    lexical_metrics: dict               # TTR, academic density, etc.
    sentence_metrics: dict              # Complex ratio, variety, etc.

    # ═══════════════════════════════════════════
    # Agent 4: Coherence & Cohesion Output (NEW)
    # ═══════════════════════════════════════════
    cc_score: float                     # Coherence & Cohesion score (0-9)
    cc_confidence: float                # Confidence of CC assessment
    cohesion_report: dict               # Devices, referencing analysis
    coherence_report: dict              # Logical progression, paragraphing
    cc_details: dict                    # Combined CC details

    # ═══════════════════════════════════════════
    # Agent 5: Chief Examiner Output
    # ═══════════════════════════════════════════
    overall_band: float                 # Final overall band score
    overall_confidence: float           # Weighted average confidence
    feedback: str                       # Detailed feedback (Markdown)
    band_breakdown: dict                # {"TA": 7.0, "CC": 6.5, ...}
    evidence_summary: dict              # Per-criterion evidence
    score_attribution: dict             # Measured strengths/weaknesses per criterion
    confidence_calibration: dict        # Per-agent confidence breakdown
    sentence_rewrites: list[dict]       # Possible alternative rewrite suggestions
    analysis_warnings: list[str]        # Components that ran in a degraded mode

    # ═══════════════════════════════════════════
    # Control Flow
    # ═══════════════════════════════════════════
    correction_count: int               # Number of correction loops completed
    needs_correction: bool              # Whether Chief flagged inconsistency
    correction_target: str              # "grounding"|"coherence"|"none"
    critic_feedback: str                # Specific feedback from Critic Agent
