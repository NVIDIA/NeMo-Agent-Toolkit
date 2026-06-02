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

from __future__ import annotations

import asyncio
import logging
import os
import time
import typing
from collections.abc import Generator

import pytest

if typing.TYPE_CHECKING:
    import uuid

    import langsmith.client

logger = logging.getLogger(__name__)


async def _wait_for_runs(
    langsmith_client: langsmith.client.Client,
    project_name: str,
    expected_count: int,
    timeout_s: float = 30.0,
) -> list:
    runs = []
    deadline = time.time() + timeout_s
    while len(runs) < expected_count and time.time() < deadline:
        runs = list(langsmith_client.list_runs(project_name=project_name))
        if len(runs) < expected_count:
            await asyncio.sleep(0.1)
    return runs


async def _wait_for_feedback(langsmith_client: langsmith.client.Client, run_ids, timeout_s: float = 15.0) -> list:
    feedback = []
    deadline = time.time() + timeout_s
    while not feedback and time.time() < deadline:
        for run_id in run_ids:
            feedback.extend(langsmith_client.list_feedback(run_ids=[run_id]))
        if not feedback:
            await asyncio.sleep(0.1)
    return feedback


@pytest.fixture(name="cleanup_prompts")
def cleanup_prompts_fixture(langsmith_client: langsmith.client.Client) -> Generator[list[str], None, None]:
    prompts: list[str] = []
    yield prompts
    if os.environ.get("NAT_CI_KEEP_LANGSMITH_PROJECTS") != "1":
        for prompt_name in prompts:
            try:
                langsmith_client.delete_prompt(prompt_name)
            except Exception:
                logger.exception("Failed to delete prompt %s", prompt_name)


@pytest.fixture(name="cleanup_datasets")
def cleanup_datasets_fixture(langsmith_client: langsmith.client.Client) -> Generator[list[uuid.UUID | str], None, None]:
    dataset_ids: list[uuid.UUID | str] = []
    yield dataset_ids
    if os.environ.get("NAT_CI_KEEP_LANGSMITH_PROJECTS") != "1":
        for dataset_id in dataset_ids:
            try:
                langsmith_client.delete_dataset(dataset_id=dataset_id)
            except Exception:
                logger.exception("Failed to delete dataset %s", dataset_id)


@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.usefixtures("langsmith_api_key")
async def test_eval_callback_creates_dataset_runs_and_feedback(
    langsmith_client: langsmith.client.Client,
    langsmith_project_name: str,
    cleanup_datasets: list[uuid.UUID | str],
):
    """Simulate a nat eval run: dataset + per-item runs + feedback."""
    from nat.eval.eval_callbacks import EvalCallbackManager
    from nat.eval.eval_callbacks import EvalResult
    from nat.eval.eval_callbacks import EvalResultItem
    from nat.eval.evaluator.evaluator_model import EvalInputItem
    from nat.plugins.langchain.langsmith.langsmith_evaluation_callback import LangSmithEvaluationCallback

    cb = LangSmithEvaluationCallback(
        project=langsmith_project_name,
        experiment_prefix="eval-integ",
    )
    mgr = EvalCallbackManager()
    mgr.register(cb)

    # 1. Load dataset
    dataset_name = f"integ-test-ds-{time.time()}"

    langsmith_client.create_run('r1', inputs={}, run_type='chain', project_name=langsmith_project_name)
    langsmith_client.create_run('r2', inputs={}, run_type='chain', project_name=langsmith_project_name)

    # Wait for runs to appear in LangSmith
    runs = await _wait_for_runs(langsmith_client, langsmith_project_name, expected_count=2)

    assert len(runs) >= 2, (f"Expected >= 2 per-item runs, got {len(runs)}")
    run_map = {run.name: run for run in runs}

    mgr.on_dataset_loaded(
        dataset_name=dataset_name,
        items=[
            EvalInputItem(
                id="q1",
                input_obj="What is 2+2?",
                expected_output_obj="4",
                full_dataset_entry={},
            ),
            EvalInputItem(
                id="q2",
                input_obj="What is 3*3?",
                expected_output_obj="9",
                full_dataset_entry={},
            ),
        ],
    )

    # Verify dataset was created with correct examples
    ds = None
    attempts = 0
    last_exception = None
    while ds is None and attempts < 5:
        try:
            ds = langsmith_client.read_dataset(dataset_name=dataset_name)
        except Exception as e:
            last_exception = e
            await asyncio.sleep(0.1)
            attempts += 1

    assert ds is not None, f"Failed to read dataset {dataset_name}: {last_exception}"
    cleanup_datasets.append(ds.id)
    examples = list(langsmith_client.list_examples(dataset_id=ds.id))
    assert len(examples) == 2
    assert {example.inputs["nat_item_id"] for example in examples} == {"q1", "q2"}

    # 2. Complete eval with per-item results
    eval_result_items = [
        EvalResultItem(
            item_id="q1",
            input_obj="What is 2+2?",
            expected_output="4",
            actual_output="4",
            scores={"accuracy": 1.0},
            reasoning={"accuracy": "Exact match"},
            root_span_id=run_map["r1"].id.int,
        ),
        EvalResultItem(
            item_id="q2",
            input_obj="What is 3*3?",
            expected_output="9",
            actual_output="8",
            scores={"accuracy": 0.8},
            reasoning={"accuracy": "Close but wrong"},
            root_span_id=run_map["r2"].id.int,
        ),
    ]
    mgr.on_eval_complete(EvalResult(
        metric_scores={"accuracy": 0.9},
        items=eval_result_items,
    ))

    # 3. Verify each eval item is linked to the run
    result_items_by_run_id = {item.root_span_id: item for item in eval_result_items}
    for run in runs:
        found_accuracy_feedback = False
        feedback = await _wait_for_feedback(langsmith_client, [run.id])
        for feedback_item in feedback:
            if feedback_item.key == "accuracy":
                result_item = result_items_by_run_id[run.id.int]
                expected_score = result_item.scores["accuracy"]
                expected_comment = result_item.reasoning["accuracy"]
                assert feedback_item.score == expected_score
                assert feedback_item.comment == expected_comment

                found_accuracy_feedback = True
        assert found_accuracy_feedback, f"No accuracy feedback found for run {run.id}"


@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.usefixtures("langsmith_api_key")
async def test_optimizer_callback_links_trial_runs_and_feedback(
    langsmith_client: langsmith.client.Client,
    langsmith_project_name: str,
    monkeypatch,
    cleanup_datasets: list[uuid.UUID | str],
):
    """Simulate optimizer OTEL runs: dataset + per-trial runs + feedback."""
    from nat.eval.eval_callbacks import EvalResult
    from nat.eval.eval_callbacks import EvalResultItem
    from nat.eval.evaluator.evaluator_model import EvalInputItem
    from nat.plugins.langchain.langsmith import langsmith_evaluation_callback
    from nat.plugins.langchain.langsmith.langsmith_optimization_callback import LangSmithOptimizationCallback
    from nat.profiler.parameter_optimization.optimizer_callbacks import OptimizerCallbackManager
    from nat.profiler.parameter_optimization.optimizer_callbacks import TrialResult

    monkeypatch.setattr(langsmith_evaluation_callback, "_LS_RETRY_DELAY_S", 0.5)

    cb = LangSmithOptimizationCallback(
        project=langsmith_project_name,
        experiment_prefix="opt-integ",
    )
    mgr = OptimizerCallbackManager()
    mgr.register(cb)

    trial_project = None
    try:
        mgr.pre_create_experiment([
            EvalInputItem(
                id="q1",
                input_obj="What is 2+2?",
                expected_output_obj="4",
                full_dataset_entry={},
            ),
            EvalInputItem(
                id="q2",
                input_obj="What is 3*3?",
                expected_output_obj="9",
                full_dataset_entry={},
            ),
        ])

        if cb._dataset_id:
            cleanup_datasets.append(cb._dataset_id)

        trial_project = mgr.get_trial_project_name(0)
        assert trial_project is not None

        langsmith_client.create_run(
            "<workflow>",
            inputs={"input": "What is 2+2?"},
            run_type="chain",
            project_name=trial_project,
        )
        langsmith_client.create_run(
            "<workflow>",
            inputs={"input": "What is 3*3?"},
            run_type="chain",
            project_name=trial_project,
        )

        runs = await _wait_for_runs(langsmith_client, trial_project, expected_count=2)
        assert len(runs) >= 2, (f"Expected >= 2 trial runs, got {len(runs)}")
        run_map = {run.inputs.get("input"): run for run in runs}

        mgr.on_trial_end(
            TrialResult(
                trial_number=0,
                parameters={"llms.nim.temperature": 0.7},
                metric_scores={"accuracy": 0.9},
                is_best=True,
                eval_result=EvalResult(
                    metric_scores={"accuracy": 0.9},
                    items=[
                        EvalResultItem(
                            item_id="q1",
                            input_obj="What is 2+2?",
                            expected_output="4",
                            actual_output="4",
                            scores={"accuracy": 1.0},
                            reasoning={"accuracy": "Exact match"},
                            root_span_id=run_map["What is 2+2?"].id.int,
                        ),
                        EvalResultItem(
                            item_id="q2",
                            input_obj="What is 3*3?",
                            expected_output="9",
                            actual_output="8",
                            scores={"accuracy": 0.8},
                            reasoning={"accuracy": "Close but wrong"},
                            root_span_id=run_map["What is 3*3?"].id.int,
                        ),
                    ],
                ),
            ))

        feedback = await _wait_for_feedback(langsmith_client, [run.id for run in runs])
        assert feedback, "No feedback found on trial runs"
        assert any(f.key == "accuracy" for f in feedback)

        mgr.on_study_end(
            best_trial=TrialResult(
                trial_number=0,
                parameters={"llms.nim.temperature": 0.7},
                metric_scores={"accuracy": 0.9},
                is_best=True,
            ),
            total_trials=1,
        )
    finally:
        if trial_project:
            try:
                langsmith_client.delete_project(project_name=trial_project)
            except Exception:
                logger.exception("Failed to delete trial project %s", trial_project)


@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.usefixtures("langsmith_api_key")
async def test_optimizer_callback_pushes_prompts(
    langsmith_client: langsmith.client.Client,
    langsmith_project_name: str,
    cleanup_prompts: list[str],
):
    """Simulate a prompt GA trial: prompts are pushed to prompt management."""
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

    repo_name = cb._prompt_repo_names["functions.agent.prompt"]
    cleanup_prompts.append(repo_name)
    assert "." not in repo_name

    prompts = []
    deadline = time.time() + 5
    while not prompts and time.time() < deadline:
        response = langsmith_client.list_prompts(query=repo_name)
        prompts = [p for p in response.repos if p.repo_handle == repo_name]
        if not prompts:
            await asyncio.sleep(0.1)

    assert prompts, f"Expected prompt repo {repo_name} to exist"
