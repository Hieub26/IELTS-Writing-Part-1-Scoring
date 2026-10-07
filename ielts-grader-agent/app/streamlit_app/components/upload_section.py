"""
Upload Section Component — File upload and essay input.
"""

from __future__ import annotations

import streamlit as st
from pathlib import Path
import tempfile
import hashlib

from app.config import settings
from app.core.utils.linguistics import count_words


def _persist_upload(uploaded_file) -> str:
    """Keep one temporary copy per session and remove a superseded upload."""
    content = uploaded_file.getvalue()
    digest = hashlib.sha256(content).hexdigest()
    previous_digest = st.session_state.get("chart_upload_digest")
    previous_path = st.session_state.get("chart_upload_path")

    if previous_digest == digest and previous_path and Path(previous_path).exists():
        return previous_path

    if previous_path:
        Path(previous_path).unlink(missing_ok=True)

    suffix = Path(uploaded_file.name).suffix.lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(content)

    st.session_state.chart_upload_digest = digest
    st.session_state.chart_upload_path = tmp.name
    return tmp.name


def render_upload_section() -> tuple[str | None, str | None, str]:
    """
    Render the upload section with chart image uploader and essay text area.

    Returns:
        Tuple of (image_path, essay_text, task_prompt); the first two are
        None until the candidate has supplied them.
    """
    st.markdown("### 📤 Upload Your Work")

    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.markdown("#### 📊 Chart / Graph Image")
        uploaded_file = st.file_uploader(
            "Upload the chart image from the exam question",
            type=["png", "jpg", "jpeg", "webp"],
            help="Supported formats: PNG, JPG, JPEG, WebP",
            key="chart_upload",
        )

        image_path = None
        if uploaded_file is not None:
            image_path = _persist_upload(uploaded_file)

            # Show preview
            st.image(uploaded_file, caption="Uploaded Chart", use_container_width=True)

        task_prompt = st.text_area(
            "Task statement (optional)",
            height=100,
            placeholder=(
                "The chart below shows ... Summarise the information by selecting "
                "and reporting the main features, and make comparisons where relevant."
            ),
            help="The question text printed with the chart. It helps the system read the chart correctly.",
            key="task_prompt_input",
        )

    with col2:
        st.markdown("#### ✍️ Your Essay")
        essay_text = st.text_area(
            "Paste your IELTS Writing Task 1 response below",
            height=350,
            placeholder=(
                "The bar chart illustrates the number of students enrolled "
                "in three different courses at a university between 2015 and 2020.\n\n"
                "Overall, it is clear that..."
            ),
            key="essay_input",
        )

        if essay_text:
            # Same counter and minimum the pipeline uses for the TA penalty.
            word_count = count_words(essay_text)
            if word_count < settings.min_word_count:
                st.warning(
                    f"⚠️ Word count: **{word_count}** "
                    f"(below {settings.min_word_count} minimum)"
                )
            else:
                st.success(f"✅ Word count: **{word_count}**")

    essay = essay_text if essay_text and essay_text.strip() else None
    return image_path, essay, (task_prompt or "").strip()
