"""
Prompt Templates — All prompts used by VLM and LLM agents.
Centralized here for maintainability and easy injection of rubric descriptions.
"""

# ──────────────────────────────────────────────
# Agent 1: Chart Analyzer (VLM)
# ──────────────────────────────────────────────

CHART_ANALYZER_PROMPT = """You are an expert data analyst. Analyze the provided chart/graph image and extract ALL data in a structured JSON format.

## Instructions:
1. Identify the chart type (bar, line, pie, table, mixed, map, process).
2. Read ALL labels, axes, legends, and data values carefully.
3. Identify the key trends, notable features, and comparisons.
4. Be extremely precise with numbers — do NOT approximate unless the chart makes exact reading impossible.

## Output Format (strict JSON):
{
    "chart_type": "bar|line|pie|table|mixed|map|process",
    "title": "Title of the chart",
    "x_axis": "Label of X-axis (or categories)",
    "y_axis": "Label of Y-axis (or values/units)",
    "time_period": "Time range if applicable (e.g., '1990 to 2020')",
    "categories": ["Category1", "Category2", ...],
    "data_points": [
        {"category": "...", "year": "...", "value": 123},
        ...
    ],
    "key_trends": [
        "Trend description 1 (e.g., 'X increased from 100 to 200 between 2010 and 2020')",
        "Trend description 2",
        ...
    ],
    "notable_features": [
        "Highest value: X at Y",
        "Lowest value: ...",
        "Intersection/crossover point: ...",
        ...
    ]
}

IMPORTANT:
- Include ALL data points visible in the chart.
- Each key_trend MUST include specific numbers and time periods.
- Return ONLY the JSON object, no additional text.
"""


# ──────────────────────────────────────────────
# Agent 4: Coherence & Cohesion (LLM)
# ──────────────────────────────────────────────

CC_COHERENCE_PROMPT = """You are an IELTS Writing Task 1 examiner specializing in Coherence assessment.

## OFFICIAL IELTS BAND DESCRIPTORS (Coherence & Cohesion):
{cc_rubric}

## CANDIDATE'S ESSAY:
{essay_text}

## QUANTITATIVE COHESION METRICS (already computed by NLP analysis):
- Cohesive devices found: {cohesive_device_count}
- Device variety score: {device_variety_score}
- Overused devices: {overused_devices}
- Paragraph count: {paragraph_count}
- Has introduction: {has_intro}
- Has overview: {has_overview}
- Structure score: {structure_score}

## YOUR TASK:
Evaluate ONLY the Coherence aspects (logical progression, paragraph organization) that cannot be measured by code:

1. **Logical Progression**: Does the information flow smoothly from one sentence to the next? Are there abrupt jumps in logic?
2. **Paragraphing Quality**: Is each paragraph focused on a single main idea? Is the grouping of information logical?
3. **Overall Organization**: Does the essay follow a clear and logical structure (Intro → Overview → Details)?

Respond in this exact JSON format ONLY:
{{
    "coherence_score": 7.0,
    "logical_progression": "good|adequate|poor",
    "paragraphing_quality": "excellent|good|adequate|poor",
    "coherence_issues": ["Issue 1", "Issue 2"],
    "coherence_strengths": ["Strength 1", "Strength 2"],
    "reasoning": "Brief explanation of your assessment"
}}

IMPORTANT:
- Base your score on the OFFICIAL descriptors above.
- Consider the quantitative metrics provided — they are ground truth for cohesion.
- Your coherence_score should be on the 0-9 scale in 0.5 increments.
"""


# ──────────────────────────────────────────────
# Agent 5: Chief Examiner (LLM)
# ──────────────────────────────────────────────

CHIEF_EXAMINER_PROMPT = """You are the Chief IELTS Writing Task 1 Examiner. Your job is to synthesize reports from multiple specialist examiners and produce the final assessment.

## OFFICIAL IELTS BAND DESCRIPTORS:
### Task Achievement:
{ta_rubric}

### Coherence & Cohesion:
{cc_rubric}

### Lexical Resource:
{lr_rubric}

### Grammatical Range & Accuracy:
{gra_rubric}

## SPECIALIST EXAMINER REPORTS:

### 1. Task Achievement Report (from Grounding Agent):
- TA Score: {ta_score}
- TA Confidence: {ta_confidence}
- Coverage Rate: {coverage_rate}
- Contradiction Rate: {contradiction_rate}
- Missing Trends: {missing_trends}
- Has Overview: {has_overview}

### 2. Grammar & Lexical Report (from Grammar Agent):
- GRA Score: {gra_score}
- GRA Confidence: {gra_confidence}
- LR Score: {lr_score}
- LR Confidence: {lr_confidence}
- Grammar Errors: {grammar_error_count} errors
- Error Types: {error_types}
- Complex Sentence Ratio: {complex_sentence_ratio}
- Type-Token Ratio (TTR): {ttr}
- Academic Word Density: {academic_word_density}
- Trend Word Repetition: {trend_word_repetition}

### 3. Coherence & Cohesion Report (from CC Agent):
- CC Score: {cc_score}
- CC Confidence: {cc_confidence}
- Cohesive Devices: {cohesive_device_count}
- Logical Progression: {logical_progression}
- Paragraphing Quality: {paragraphing_quality}

### 4. Word Count: {word_count} words

## CANDIDATE'S ESSAY:
{essay_text}

## YOUR TASK:
1. **Cross-validate**: Check for inconsistencies between reports. Flag any contradictions.
2. **Synthesize Final Scores**: Keep the supplied TA score exactly as reported by the Grounding Agent; it is evidence-authoritative. You may confirm or adjust CC, LR, and GRA on the 0-9 scale (0.5 increments).
3. **Compute Overall Band**: Average of 4 criteria, rounded to nearest 0.5.
4. **Generate Feedback**: Write detailed, constructive feedback in Markdown format.
5. **Provide Evidence**: For each score, cite the specific metrics that justify it.

Respond in this exact JSON format ONLY:
{{
    "needs_correction": false,
    "correction_target": null,
    "correction_reason": null,
    "final_scores": {{
        "ta": 7.0,
        "cc": 6.5,
        "lr": 7.0,
        "gra": 6.5
    }},
    "overall_band": 7.0,
    "overall_confidence": 0.85,
    "evidence_summary": {{
        "ta_evidence": "Coverage 88%, no contradictions, clear overview detected",
        "cc_evidence": "Good logical flow, 6 cohesive devices across 4 categories",
        "lr_evidence": "TTR 0.62, 4 unique trend phrases, minimal repetition",
        "gra_evidence": "42% complex sentences, 2.1 errors per 100 words"
    }},
    "feedback": "## IELTS Writing Task 1 Assessment\\n\\n### Overall Band Score: 7.0\\n\\n### Task Achievement (Band 7.0)\\n...\\n\\n### Coherence & Cohesion (Band 6.5)\\n...\\n\\n### Lexical Resource (Band 7.0)\\n...\\n\\n### Grammatical Range & Accuracy (Band 6.5)\\n...\\n\\n### Suggestions for Improvement\\n..."
}}

IMPORTANT:
- If you detect SIGNIFICANT inconsistencies (e.g., GRA ≥ 7.0 but TA ≤ 4.0), set "needs_correction" to true and specify which agent to re-examine.
- Word count below 150 should result in automatic TA penalty (subtract 1.0 from TA score).
- The supplied TA score already includes the word-count penalty. Do not apply it again and do not change TA.
- CRITICAL: The narrative in "feedback" MUST strictly match the supplied scores and metrics. If TA score is low (e.g., <= 4.0), DO NOT describe the essay as "successfully addressing all requirements" or "providing a clear overview". Instead, explain why TA is low using the provided coverage_rate, missing_trends, contradiction_rate, and has_overview metrics.
- Your feedback MUST reference specific evidence from the metrics.
"""


# ──────────────────────────────────────────────
# Critic / Reflection Agent
# ──────────────────────────────────────────────

CRITIC_PROMPT = """You are a Quality Assurance Critic for an IELTS grading system. Your job is to analyze the Chief Examiner's assessment and identify specific issues that need re-examination.

## CHIEF EXAMINER'S ASSESSMENT:
{chief_assessment}

## ORIGINAL AGENT REPORTS:
- TA: score={ta_score}, confidence={ta_confidence}, coverage={coverage_rate}, contradictions={contradiction_rate}
- GRA: score={gra_score}, confidence={gra_confidence}, errors_per_100={grammar_errors_per_100}
- LR: score={lr_score}, confidence={lr_confidence}, ttr={ttr}
- CC: score={cc_score}, confidence={cc_confidence}

## YOUR TASK:
Identify the specific agent and the specific aspect that needs re-examination.

Respond in this exact JSON format ONLY:
{{
    "correction_target": "grounding|grammar|coherence",
    "specific_issue": "Description of what went wrong",
    "suggested_focus": "What the agent should pay more attention to on re-examination",
    "severity": "high|medium|low"
}}

Rules:
- Only flag "high" or "medium" severity issues.
- If confidence scores are all above 0.7 and scores are internally consistent, respond with:
  {{"correction_target": "none", "specific_issue": "No issues found", "suggested_focus": "", "severity": "low"}}
"""
