"""Measure grading accuracy against essays with known examiner bands.

Usage: python scripts/evaluate.py data/eval/labels.csv

The CSV needs a header row with these columns (paths are relative to the CSV):

    image,essay,ta,cc,lr,gra,overall[,task_prompt]

``image`` is the chart image, ``essay`` a UTF-8 text file, and the band
columns hold the examiner's scores.  Run this before and after changing any
threshold, prompt or rubric heuristic: a change is an improvement only if the
error against real examiner scores goes down.

This grades every row through the real pipeline, so it makes Gemini requests.
"""

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.graph import grade_essay

CRITERIA = {"ta": "TA", "cc": "CC", "lr": "LR", "gra": "GRA"}


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__)
        return 2

    labels_path = Path(sys.argv[1]).resolve()
    with open(labels_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        print(f"No rows found in {labels_path}")
        return 1

    errors = {key: [] for key in [*CRITERIA, "overall"]}
    failed = 0

    for index, row in enumerate(rows, start=1):
        image = labels_path.parent / row["image"]
        essay = (labels_path.parent / row["essay"]).read_text(encoding="utf-8")
        result = grade_essay(str(image), essay, task_prompt=row.get("task_prompt") or "")

        if result.get("chart_analysis_error"):
            failed += 1
            print(f"[{index}/{len(rows)}] {row['essay']}: chart analysis failed, skipped")
            continue

        predicted = {
            **{key: result["band_breakdown"][name] for key, name in CRITERIA.items()},
            "overall": result["overall_band"],
        }
        for key, value in predicted.items():
            errors[key].append(value - float(row[key]))
        print(
            f"[{index}/{len(rows)}] {row['essay']}: "
            f"overall {predicted['overall']} (examiner {row['overall']})"
        )

    graded = len(rows) - failed
    if not graded:
        print("No essay could be graded.")
        return 1

    print(f"\nGraded {graded} of {len(rows)} essays")
    print(f"{'criterion':<10}{'MAE':>6}{'bias':>7}{'within 0.5':>12}")
    for key, diffs in errors.items():
        mae = sum(abs(d) for d in diffs) / graded
        bias = sum(diffs) / graded
        within = sum(abs(d) <= 0.5 for d in diffs) / graded
        print(f"{key:<10}{mae:>6.2f}{bias:>+7.2f}{within:>12.0%}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
