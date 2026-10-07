# IELTS Multi-Agent Grader

Streamlit application for estimating IELTS Writing Task 1 bands from a chart image and an essay.

## Setup

1. Create and activate a Python 3.10+ virtual environment.
2. Install packages: `python -m pip install -r requirements.txt`.
3. Install the spaCy model: `python -m spacy download en_core_web_sm`.
4. Install a Java runtime (Java 17+ recommended). LanguageTool uses Java for grammar checks; if unavailable, the app continues with reduced grammar analysis and says so in the result.
5. Copy `.env.example` to `.env`, then set a valid `GEMINI_API_KEY`.

Run the app with `python run.py` (add `--port 8600` to change the port).

Gemini is required to read the uploaded chart. The application gives a clear configuration error when its API key is absent instead of attempting an invalid request.

## Verification

- `python -m pytest -q` runs the test suite. Gemini, the NLI model and LanguageTool are mocked, so it needs no API key and makes no network requests.
- `python scripts/check_gemini.py` makes one real request to confirm the API key and model names work.

## Measuring scoring accuracy

The band thresholds in `rubrics/` are heuristics. They have not yet been validated against examiner-scored essays, so treat every band as an estimate.

To measure and improve them, collect essays with known examiner bands and list them in a CSV:

```
image,essay,ta,cc,lr,gra,overall
charts/energy.png,essays/energy_01.txt,7,7,6.5,7,7
```

Then run `python scripts/evaluate.py path/to/labels.csv`. It grades every row through the real pipeline and reports, per criterion, the mean absolute error, the bias, and the share of essays scored within half a band. Re-run it after changing any threshold or prompt and keep the change only if the error goes down.
