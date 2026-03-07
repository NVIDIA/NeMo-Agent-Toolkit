# A minimal BYOB benchmark for testing.
# Uses exact_match scorer with a tiny inline JSONL dataset.

from nemo_evaluator.contrib.byob import ScorerInput, benchmark, scorer
from nemo_evaluator.contrib.byob.scorers import exact_match

import json
import os
import tempfile

# Create a tiny test dataset as a temp file
_DATA = [
    {"id": "0", "question": "What is 2+2?", "target": "4"},
    {"id": "1", "question": "What color is the sky?", "target": "blue"},
    {"id": "2", "question": "Capital of France?", "target": "Paris"},
]

_DATASET_PATH = os.path.join(tempfile.gettempdir(), "byob_test_dataset.jsonl")
with open(_DATASET_PATH, "w") as f:
    for row in _DATA:
        f.write(json.dumps(row) + "\n")


@benchmark(
    name="test-exact-match",
    dataset=_DATASET_PATH,
    prompt="{question}",
    target_field="target",
)
@scorer
def test_exact_match_scorer(sample: ScorerInput) -> dict:
    return exact_match(sample)
