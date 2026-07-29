# IELTS Multi-Agent Grader

Streamlit application for estimating IELTS Writing Task 1 bands from a chart image and an essay.

## Setup

1. Create and activate a Python 3.10+ virtual environment.
2. Install packages: `python -m pip install -r requirements.txt`.
3. Install the spaCy model: `python -m spacy download en_core_web_sm`.
4. Install a Java runtime (Java 17+ recommended). LanguageTool uses Java for grammar checks; if unavailable, the app continues with reduced grammar analysis.
5. Copy `.env.example` to `.env`, then set a valid `GEMINI_API_KEY`.

Run the app with `python run.py`. Run verification with `python -m pytest -q`.

Gemini is required to read the uploaded chart. The application gives a clear configuration error when its API key is absent instead of attempting an invalid request.
