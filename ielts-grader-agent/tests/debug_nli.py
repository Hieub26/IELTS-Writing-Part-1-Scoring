"""Debug: test FIXED bidirectional NLI matching."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from app.core.utils.linguistics import segment_sentences, resolve_coreferences

# Load essay
with open("data/samples/employment-town-sample.txt", "r") as f:
    essay = f.read()

sents = segment_sentences(essay)
resolved = resolve_coreferences(sents)

print(f"Sentences: {len(sents)}")
for i, s in enumerate(sents):
    print(f"  [{i}] {s[:100]}")

# Load NLI model
print("\nLoading NLI model...")
from sentence_transformers import CrossEncoder
model = CrossEncoder("cross-encoder/nli-deberta-v3-base")

def softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum()

NLI_LABELS = {0: "contradiction", 1: "entailment", 2: "neutral"}

# Simulated chart trends
trends = [
    "Manufacturing employment in Town A increased from 20% in 2009 to 60% in 2020",
    "Engineering in Town A decreased from 60% in 2009 to 17% in 2020",
    "Hospitality in Town A increased slightly from 11% to 15%",
    "Teaching in Town A decreased from 9% to 8%",
    "Manufacturing in Town B decreased from 40% to 25%",
    "Teaching in Town B doubled from 10% to 20%",
    "Hospitality in Town B increased from 15% to 20%",
    "Engineering in Town B remained unchanged at 35%",
]

print("\n" + "=" * 60)
print("BIDIRECTIONAL NLI (FIXED)")
print("=" * 60)

entailed_count = 0

for trend_idx, trend in enumerate(trends):
    print(f"\nTrend [{trend_idx}]: {trend}")
    best_label = "neutral"
    best_score = 0.0
    best_sent = -1

    for sent_idx, resolved_sent in enumerate(resolved):
        context = resolved_sent[:400]

        # Forward: (essay -> trend)
        fwd_raw = model.predict([(context, trend)])[0]
        fwd_probs = softmax(fwd_raw)

        # Reverse: (trend -> essay)
        rev_raw = model.predict([(trend, context)])[0]
        rev_probs = softmax(rev_raw)

        # Best entailment from either direction
        fwd_ent = fwd_probs[1]
        rev_ent = rev_probs[1]
        entail = max(fwd_ent, rev_ent)
        direction = "fwd" if fwd_ent >= rev_ent else "rev"

        if entail > 0.3:  # Print anything remotely interesting
            print(f"  Sent [{sent_idx}] entail={entail:.3f} ({direction}) | fwd={fwd_ent:.3f} rev={rev_ent:.3f}")

        if entail > 0.5 and entail > best_score:
            best_label = "entailment"
            best_score = float(entail)
            best_sent = sent_idx

    if best_label == "entailment":
        entailed_count += 1
    print(f"  >>> RESULT: {best_label} (score={best_score:.3f}, sent={best_sent})")

coverage = entailed_count / len(trends)
print(f"\n{'=' * 60}")
print(f"COVERAGE: {entailed_count}/{len(trends)} = {coverage:.0%}")
print(f"{'=' * 60}")
