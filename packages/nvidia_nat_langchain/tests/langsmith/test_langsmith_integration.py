# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Integration tests for LangSmith eval and optimizer callbacks.

These tests exercise the callback -> LangSmith SDK flow with a real API key.
They create real datasets, runs, and feedback in LangSmith and verify the
results via the LangSmith client.

Requirements:
    - LANGSMITH_API_KEY environment variable must be set
    - Network access to LangSmith API
    - nvidia-nat-test package installed (provides test fixtures)

Run with:
    pytest packages/nvidia_nat_langchain/tests/langsmith/test_langsmith_integration.py \
        --run_integration --run_slow -v

Tests are skipped by default. Use --run_integration and --run_slow to enable.
"""

import asyncio
import time

import pytest


@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.usefixtures("langsmith_api_key")
async def test_eval_callback_creates_dataset_runs_and_feedback(
    langsmith_client,
    langsmith_project_name: str,
):
    """Simulate a nat eval run: dataset + per-item runs + feedback."""
    from nat.eval.eval_callbacks import EvalCallbackManager
    from nat.eval.eval_callbacks import EvalResult
    from nat.eval.eval_callbacks import EvalResultItem
    from nat.plugins.langchain.langsmith.langsmith_evaluation_callback import LangSmithEvaluationCallback

    cb = LangSmithEvaluationCallback(
        project=langsmith_project_name,
        experiment_prefix="eval-integ",
    )
    mgr = EvalCallbackManager()
    mgr.register(cb)

    # 1. Load dataset
    dataset_name = f"integ-test-ds-{time.time()}"
    mgr.on_dataset_loaded(
        dataset_name=dataset_name,
        items=[
            {
                "id": "q1",
                "question": "What is 2+2?",
                "expected_output": "4",
            },
            {
                "id": "q2",
                "question": "What is 3*3?",
                "expected_output": "9",
            },
        ],
    )

    # Verify dataset was created with correct examples
    ds = langsmith_client.read_dataset(dataset_name=dataset_name)
    assert ds is not None
    examples = list(langsmith_client.list_examples(dataset_id=ds.id))
    assert len(examples) == 2

    # 2. Complete eval with per-item results
    mgr.on_eval_complete(
        EvalResult(
            metric_scores={"accuracy": 0.9},
            items=[
                EvalResultItem(
                    item_id="q1",
                    input_obj="What is 2+2?",
                    expected_output="4",
                    actual_output="4",
                    scores={"accuracy": 1.0},
                    reasoning={"accuracy": "Exact match"},
                ),
                EvalResultItem(
                    item_id="q2",
                    input_obj="What is 3*3?",
                    expected_output="9",
                    actual_output="8",
                    scores={"accuracy": 0.8},
                    reasoning={"accuracy": "Close but wrong"},
                ),
            ],
        ))

    # 3. Wait for runs to appear in LangSmith
    runs = []
    deadline = time.time() + 15
    while len(runs) < 2 and time.time() < deadline:
        await asyncio.sleep(1)
        runs = list(langsmith_client.list_runs(project_name=langsmith_project_name, ))

    assert len(runs) >= 2, (f"Expected >= 2 per-item runs, got {len(runs)}")

    # 4. Verify feedback was attached to at least one run
    feedback_found = False
    for run in runs:
        fb = list(langsmith_client.list_feedback(run_ids=[run.id]))
        if fb:
            feedback_found = True
            assert any(f.key == "accuracy" for f in fb)
            break
    assert feedback_found, "No feedback found on any run"

    # Cleanup: delete the dataset we created
    langsmith_client.delete_dataset(dataset_id=ds.id)


@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.usefixtures("langsmith_api_key")
async def test_optimizer_callback_creates_trial_runs_and_summary(
    langsmith_client,
    langsmith_project_name: str,
):
    """Simulate optimizer trials: trial runs + study summary + feedback."""
    from nat.plugins.langchain.langsmith.langsmith_optimization_callback import LangSmithOptimizationCallback
    from nat.profiler.parameter_optimization.optimizer_callbacks import OptimizerCallbackManager
    from nat.profiler.parameter_optimization.optimizer_callbacks import TrialResult

    cb = LangSmithOptimizationCallback(
        project=langsmith_project_name,
        experiment_prefix="opt-integ",
    )
    mgr = OptimizerCallbackManager()
    mgr.register(cb)

    for i in range(2):
        mgr.on_trial_end(
            TrialResult(
                trial_number=i,
                parameters={"llms.nim.temperature": 0.5 + i * 0.2},
                metric_scores={"accuracy": 0.8 + i * 0.05},
                is_best=(i == 1),
            ))

    mgr.on_study_end(
        best_trial=TrialResult(
            trial_number=1,
            parameters={"llms.nim.temperature": 0.7},
            metric_scores={"accuracy": 0.85},
            is_best=True,
        ),
        total_trials=2,
    )

    try:
        # Wait for runs to appear: 2 trial runs + 1 summary = 3
        runs = []
        deadline = time.time() + 15
        while len(runs) < 3 and time.time() < deadline:
            await asyncio.sleep(1)
            runs = list(langsmith_client.list_runs(project_name=langsmith_project_name, ))

        assert len(runs) >= 3, (f"Expected >= 3 runs (2 trials + 1 summary), got {len(runs)}")

        # Verify the summary run exists and has the correct outputs
        summary_runs = [r for r in runs if "summary" in (r.name or "")]
        assert len(summary_runs) >= 1, ("Expected at least 1 summary run")
        assert summary_runs[0].outputs.get("best_trial_number") == 1
    finally:
        try:
            langsmith_client.delete_project(project_name=langsmith_project_name)
        except Exception:
            pass


@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.usefixtures("langsmith_api_key")
async def test_optimizer_callback_pushes_prompts(
    langsmith_client,
    langsmith_project_name: str,
):
    """Simulate a prompt GA trial: prompts in run inputs + pushed to
    prompt management."""
    from nat.plugins.langchain.langsmith.langsmith_optimization_callback import LangSmithOptimizationCallback
    from nat.profiler.parameter_optimization.optimizer_callbacks import OptimizerCallbackManager
    from nat.profiler.parameter_optimization.optimizer_callbacks import TrialResult

    cb = LangSmithOptimizationCallback(
        project=langsmith_project_name,
        experiment_prefix="prompt-integ",
    )
    mgr = OptimizerCallbackManager()
    mgr.register(cb)

    mgr.on_trial_end(
        TrialResult(
            trial_number=0,
            parameters={},
            metric_scores={"accuracy": 0.9},
            is_best=True,
            prompts={
                "functions.agent.prompt": ("You are a helpful math assistant."),
            },
        ))

    try:
        # Wait for the run to appear in LangSmith
        runs = []
        deadline = time.time() + 15
        while len(runs) < 1 and time.time() < deadline:
            await asyncio.sleep(1)
            runs = list(langsmith_client.list_runs(project_name=langsmith_project_name, ))

        assert len(runs) >= 1, (f"Expected >= 1 run, got {len(runs)}")

        # Verify prompts are included in the run inputs
        assert "prompts" in runs[0].inputs, ("Expected 'prompts' key in run inputs")
        assert "functions.agent.prompt" in runs[0].inputs["prompts"], ("Expected prompt param name in run inputs")
    finally:
        try:
            langsmith_client.delete_project(project_name=langsmith_project_name)
        except Exception:
            pass
