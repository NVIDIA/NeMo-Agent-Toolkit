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
"""Integration test for nested trajectory integrity in simple calculator eval."""

import json
from pathlib import Path

import pytest

from nat.data_models.evaluate_runtime import EvaluationRunConfig
from nat.plugins.eval.runtime.evaluate import EvaluationRun
from nat.test.utils import locate_example_config


@pytest.mark.integration
@pytest.mark.usefixtures("nvidia_api_key")
async def test_nested_trajectory_eval_emits_power_of_two_and_multiply(tmp_path: Path):
    """Ensure nested tool lineage is preserved in ATIF workflow output."""
    import nat_simple_calculator_eval

    config_file: Path = locate_example_config(nat_simple_calculator_eval, "config-nested-trajectory-eval.yml")
    output_dir = tmp_path / "nested-trajectory-eval"

    config = EvaluationRunConfig(
        config_file=config_file,
        dataset=None,
        result_json_path="$",
        skip_workflow=False,
        skip_completed_entries=False,
        endpoint=None,
        endpoint_timeout=30,
        reps=1,
        override=(
            ("eval.general.max_concurrency", "1"),
            ("eval.general.output.dir", str(output_dir)),
        ),
    )

    eval_runner = EvaluationRun(config=config)
    output = await eval_runner.run_and_evaluate()

    assert not output.workflow_interrupted, "The workflow was interrupted"
    assert output.workflow_output_file, "The workflow_output.json file was not created"
    assert output.workflow_output_file.exists(), "The workflow_output.json file was not created"

    atif_workflow_output = output.workflow_output_file.parent / "workflow_output_atif.json"
    assert atif_workflow_output.exists(), "The workflow_output_atif.json file was not created"

    trajectory_eval_output: Path | None = None
    for output_file in output.evaluator_output_files:
        if "trajectory_eval_output" in str(output_file):
            trajectory_eval_output = output_file
            break
    assert trajectory_eval_output and trajectory_eval_output.exists(), "The trajectory evaluator output was not created"

    payload = json.loads(atif_workflow_output.read_text(encoding="utf-8"))
    assert isinstance(payload, list) and payload, "ATIF workflow output is empty"

    saw_power_of_two_tool_call = False
    saw_power_of_two_in_tool_ancestry = False
    saw_calculator_multiply_in_tool_ancestry = False

    for item in payload:
        if not isinstance(item, dict):
            continue
        trajectory = item.get("trajectory")
        if not isinstance(trajectory, dict):
            continue
        for step in trajectory.get("steps", []):
            if not isinstance(step, dict):
                continue
            for tool_call in step.get("tool_calls") or []:
                if isinstance(tool_call, dict) and tool_call.get("function_name") == "power_of_two":
                    saw_power_of_two_tool_call = True

            extra = step.get("extra") or {}
            for tool_ancestry in extra.get("tool_ancestry") or []:
                if not isinstance(tool_ancestry, dict):
                    continue
                fn = tool_ancestry.get("function_ancestry")
                if not isinstance(fn, dict):
                    continue
                name = fn.get("function_name")
                parent_name = fn.get("parent_name")
                if name == "power_of_two" or parent_name == "power_of_two":
                    saw_power_of_two_in_tool_ancestry = True
                if name == "calculator__multiply":
                    saw_calculator_multiply_in_tool_ancestry = True

    assert saw_power_of_two_tool_call, "Expected at least one tool call to power_of_two"
    assert saw_power_of_two_in_tool_ancestry, "Expected power_of_two in tool_ancestry lineage"
    assert saw_calculator_multiply_in_tool_ancestry, "Expected calculator__multiply in tool_ancestry lineage"
