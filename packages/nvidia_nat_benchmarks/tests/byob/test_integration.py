# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Integration tests for the BYOB evaluator pipeline.

Tests the evaluator directly (no nat eval subprocess needed) using a minimal
BYOB benchmark definition with pre-constructed eval items.
No LLM or network access required — this validates the BYOB scoring pipeline.
"""

import json
import os

import pytest

try:
    from nemo_evaluator.contrib.byob.eval_logic import import_benchmark
    _HAS_BYOB = True
except ImportError:
    _HAS_BYOB = False

pytestmark = pytest.mark.skipif(not _HAS_BYOB, reason="nemo_evaluator BYOB not installed")

from nat.data_models.evaluator import EvalInputItem  # noqa: E402
from nat.plugins.benchmarks.byob.evaluator import _evaluate_single  # noqa: E402


class TestBYOBIntegration:

    def test_byob_scorer_pipeline_exact_match(self):
        """Full pipeline: import benchmark → create items → score with exact_match."""
        fixture_path = os.path.join(os.path.dirname(__file__), "fixtures", "sample_benchmark.py")
        bench = import_benchmark(fixture_path, "test_exact_match")

        # Simulate items that the eval runner would create
        items = [
            EvalInputItem(
                id="0",
                input_obj=json.dumps({
                    "question": "What is 2+2?", "target": "4"
                }),
                expected_output_obj="4",
                output_obj="4",  # Correct answer
                full_dataset_entry={
                    "question": "What is 2+2?", "target": "4"
                },
            ),
            EvalInputItem(
                id="1",
                input_obj=json.dumps({
                    "question": "Sky color?", "target": "blue"
                }),
                expected_output_obj="blue",
                output_obj="red",  # Wrong answer
                full_dataset_entry={
                    "question": "Sky color?", "target": "blue"
                },
            ),
            EvalInputItem(
                id="2",
                input_obj=json.dumps({
                    "question": "Capital of France?", "target": "Paris"
                }),
                expected_output_obj="Paris",
                output_obj="Paris",  # Correct answer
                full_dataset_entry={
                    "question": "Capital of France?", "target": "Paris"
                },
            ),
        ]

        results = []
        for item in items:
            result = _evaluate_single(
                item,
                bench.scorer_fn,
                bench.target_field,
                "correct",
                bench.extra_config,
            )
            results.append(result)

        # 2 correct, 1 wrong → scores [1.0, 0.0, 1.0]
        assert results[0].score == 1.0
        assert results[1].score == 0.0
        assert results[2].score == 1.0

        avg = sum(r.score for r in results) / len(results)
        assert abs(avg - 2 / 3) < 0.01

    def test_byob_dataset_to_evaluator_roundtrip(self):
        """Full roundtrip: load dataset → create items → score."""
        from nat.plugins.benchmarks.byob.dataset import load_byob_dataset

        fixture_path = os.path.join(os.path.dirname(__file__), "fixtures", "sample_benchmark.py")

        # Load dataset
        df = load_byob_dataset("ignored", benchmark_module=fixture_path, benchmark_name="test_exact_match")
        assert len(df) == 3

        # Import benchmark for scorer
        bench = import_benchmark(fixture_path, "test_exact_match")

        # Create eval items from DataFrame (simulating what DatasetHandler does)
        items = []
        for _, row in df.iterrows():
            items.append(
                EvalInputItem(
                    id=row["id"],
                    input_obj=row["question"],
                    expected_output_obj=row["answer"],
                    output_obj=row["answer"],  # Simulate perfect answers
                    full_dataset_entry=json.loads(row["question"]),
                ))

        # Score all items
        results = []
        for item in items:
            result = _evaluate_single(
                item,
                bench.scorer_fn,
                bench.target_field,
                "correct",
                bench.extra_config,
            )
            results.append(result)

        # All should be correct (output == expected)
        assert all(r.score == 1.0 for r in results)
