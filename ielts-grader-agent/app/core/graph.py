"""
LangGraph Orchestration — Defines the Multi-Agent grading pipeline.

Graph Architecture:
  START → preprocess → chart_analyzer → [chart_error_router] →
    → parallel fan-out: [grounding, grammar_lexical, coherence] →
    → fan-in join → chief_examiner →
    → [correction_router]:
        needs_correction? → critic → [correction_target_router]:
            → re-run grounding or coherence → chief_examiner (loop)
        no correction → END

Features:
  - Early exit when the chart cannot be read
  - Parallel execution of scoring agents (TA, GRA/LR, CC)
  - Self-correction loop with Critic/Reflection pattern
  - Maximum loop protection

Every chart type follows the same path; map and process diagrams are handled
by the chart analyzer describing changes/stages instead of numeric trends.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Callable

from langgraph.graph import StateGraph, START, END

from app.core.state import GraderState
from app.core.utils.linguistics import count_words

# Import agent nodes
from app.core.agents.chart_analyzer import chart_analyzer_node
from app.core.agents.grounding import grounding_node
from app.core.agents.grammar_lexical import grammar_lexical_node
from app.core.agents.coherence import coherence_node
from app.core.agents.chief_examiner import chief_examiner_node
from app.core.agents.critic import critic_node


# ──────────────────────────────────────────────
# Preprocessing Node
# ──────────────────────────────────────────────

def preprocess_node(state: GraderState) -> dict:
    """
    Initialize state with computed values.
    Runs before any agent to set up word count and defaults.
    """
    essay_text = state.get("essay_text", "")
    return {
        "word_count": count_words(essay_text),
        "correction_count": 0,
        "needs_correction": False,
        "correction_target": "none",
        "critic_feedback": "",
    }


def fan_out_node(_: GraderState) -> dict:
    """Explicit fork point for the three independent scoring agents."""
    return {}


# ──────────────────────────────────────────────
# Conditional Edge Functions
# ──────────────────────────────────────────────

def route_after_chart(state: GraderState) -> str:
    """Route after chart analysis — skip grading if chart extraction failed."""
    if state.get("chart_analysis_error"):
        return "error_end"
    return "parallel_grade"


def route_after_chief(state: GraderState) -> str:
    """Route after Chief Examiner — enter correction loop or finish."""
    if state.get("needs_correction", False):
        return "critic"
    return "end"


def route_after_critic(state: GraderState) -> str:
    """Route after Critic — direct to the specific agent that needs re-examination."""
    target = state.get("correction_target", "none")
    if not state.get("needs_correction", False):
        return "end"
    if target == "grounding":
        return "re_grounding"
    elif target == "coherence":
        return "re_coherence"
    else:
        return "end"


# ──────────────────────────────────────────────
# Error Handling Node
# ──────────────────────────────────────────────

def error_end_node(state: GraderState) -> dict:
    """Handle chart extraction errors gracefully."""
    error = state.get("chart_analysis_error", "Unknown error")
    return {
        "overall_band": 0.0,
        "overall_confidence": 0.0,
        "feedback": (
            f"## ❌ Chart Analysis Failed\n\n"
            f"The system could not extract data from the chart image.\n\n"
            f"**Error**: {error}\n\n"
            f"Please try again with a clearer image."
        ),
        "band_breakdown": {"TA": 0, "CC": 0, "LR": 0, "GRA": 0},
        "evidence_summary": {},
        "needs_correction": False,
    }


# ──────────────────────────────────────────────
# Graph Builder
# ──────────────────────────────────────────────

def build_grading_graph() -> StateGraph:
    """
    Build and compile the LangGraph grading pipeline.

    Architecture:
        preprocess → chart_analyzer → [route]
            → error_end (if chart failed)
            → parallel: grounding + grammar_lexical + coherence
                → chief_examiner → [route]
                    → critic → [route by target] → re-run grounding/coherence → chief_examiner
                    → END
    """
    builder = StateGraph(GraderState)

    # ═══════════════════════════════════════════
    # Add Nodes
    # ═══════════════════════════════════════════

    builder.add_node("preprocess", preprocess_node)
    builder.add_node("chart_analyzer", chart_analyzer_node)
    builder.add_node("fan_out", fan_out_node)
    builder.add_node("error_end", error_end_node)

    # Parallel scoring agents
    builder.add_node("grounding", grounding_node)
    builder.add_node("grammar_lexical", grammar_lexical_node)
    builder.add_node("coherence", coherence_node)

    # Synthesis & correction
    builder.add_node("chief_examiner", chief_examiner_node)
    builder.add_node("critic", critic_node)

    # Re-run nodes (same functions, different node names for graph clarity).
    # Grammar/lexical analysis is deterministic, so it has no re-run node.
    builder.add_node("re_grounding", grounding_node)
    builder.add_node("re_coherence", coherence_node)

    # ═══════════════════════════════════════════
    # Add Edges
    # ═══════════════════════════════════════════

    # Entry: START → preprocess → chart_analyzer
    builder.add_edge(START, "preprocess")
    builder.add_edge("preprocess", "chart_analyzer")

    # After chart analysis: conditional routing
    builder.add_conditional_edges(
        "chart_analyzer",
        route_after_chart,
        {
            "error_end": "error_end",
            "parallel_grade": "fan_out",
        },
    )

    # Error end → END
    builder.add_edge("error_end", END)

    # Fan out to independent scoring agents.  LangGraph waits for every
    # predecessor of ``chief_examiner`` before scheduling it, creating a
    # proper fan-in without serialising the three analyses.
    builder.add_edge("fan_out", "grounding")
    builder.add_edge("fan_out", "grammar_lexical")
    builder.add_edge("fan_out", "coherence")
    builder.add_edge("grounding", "chief_examiner")
    builder.add_edge("grammar_lexical", "chief_examiner")
    builder.add_edge("coherence", "chief_examiner")

    # After Chief Examiner: correction loop or END
    builder.add_conditional_edges(
        "chief_examiner",
        route_after_chief,
        {
            "critic": "critic",
            "end": END,
        },
    )

    # After Critic: route to specific agent for re-examination
    builder.add_conditional_edges(
        "critic",
        route_after_critic,
        {
            "re_grounding": "re_grounding",
            "re_coherence": "re_coherence",
            "end": END,
        },
    )

    # Re-run agents feed back to Chief Examiner
    builder.add_edge("re_grounding", "chief_examiner")
    builder.add_edge("re_coherence", "chief_examiner")

    return builder


@lru_cache(maxsize=1)
def compile_graph():
    """Build and compile the grading graph."""
    builder = build_grading_graph()
    return builder.compile()


# ──────────────────────────────────────────────
# Convenience Runner
# ──────────────────────────────────────────────

def grade_essay(
    image_path: str,
    essay_text: str,
    task_prompt: str = "",
    on_node_complete: Callable[[str], None] | None = None,
) -> dict:
    """
    Grade an IELTS Writing Task 1 essay.

    Args:
        image_path: Path to the chart/graph image.
        essay_text: The candidate's essay text.
        task_prompt: Optional task statement printed with the chart.
        on_node_complete: Called with a node's name each time it finishes,
            so a UI can report real pipeline progress.

    Returns:
        Final GraderState with all scores, feedback, and evidence.
    """
    graph = compile_graph()
    initial_state = {
        "image_path": image_path,
        "essay_text": essay_text,
        "task_prompt": task_prompt,
        "word_count": 0,
        "chart_type": "",
        "chart_data": {},
        "chart_analysis_error": "",
        "ta_score": 0.0,
        "ta_confidence": 0.0,
        "ta_details": {},
        "grounding_report": [],
        "gra_score": 0.0,
        "gra_confidence": 0.0,
        "lr_score": 0.0,
        "lr_confidence": 0.0,
        "grammar_errors": [],
        "lexical_metrics": {},
        "sentence_metrics": {},
        "cc_score": 0.0,
        "cc_confidence": 0.0,
        "cohesion_report": {},
        "coherence_report": {},
        "cc_details": {},
        "overall_band": 0.0,
        "overall_confidence": 0.0,
        "feedback": "",
        "band_breakdown": {},
        "evidence_summary": {},
        "analysis_warnings": [],
        "correction_count": 0,
        "needs_correction": False,
        "correction_target": "none",
        "critic_feedback": "",
    }

    result = initial_state
    for mode, chunk in graph.stream(initial_state, stream_mode=["updates", "values"]):
        if mode == "values":
            result = chunk
        elif on_node_complete:
            for node_name in chunk:
                on_node_complete(node_name)
    return result
