# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import importlib.resources
import inspect
import json
import logging
from pathlib import Path

import pytest

from nat.eval.evaluate import EvaluationRun
from nat.eval.evaluate import EvaluationRunConfig
from nat_swe_bench.register import SweBenchWorkflowConfig

logger = logging.getLogger(__name__)


def validate_workflow_output(workflow_output_file: Path):
    """
    Validate the contents of the workflow output file.
    WIP: output format should be published as a schema and this validation should be done against that schema.
    """
    # Ensure the workflow_output.json file was created
    assert workflow_output_file.exists(), "The workflow_output.json file was not created"

    # Read and validate the workflow_output.json file
    try:
        with open(workflow_output_file, "r", encoding="utf-8") as f:
            result_json = json.load(f)
    except json.JSONDecodeError:
        pytest.fail("Failed to parse workflow_output.json as valid JSON")

    assert isinstance(result_json, list), "The workflow_output.json file is not a list"
    assert len(result_json) > 0, "The workflow_output.json file is empty"
    assert isinstance(result_json[0], dict), "The workflow_output.json file is not a list of dictionaries"

    # Ensure required keys exist
    required_keys = ["instance_id", "model_name_or_path", "model_patch"]
    for key in required_keys:
        assert all(item.get(key) for item in result_json), f"The '{key}' key is missing in workflow_output.json"


def validate_evaluation_output(eval_output_file: Path):
    """
    1. Validate the contents of the swe_bench_output.json file.
    2. Ensure the average_accuracy is above a minimum threshold.
    WIP: output format should be published as a schema and this validation should be done against that schema.
    """
    # we are using golden patches so the score should be perfect
    score_min = 1.0
    # Ensure the file exists
    assert eval_output_file and eval_output_file.exists(), \
        f"The {eval_output_file} file was not created"
    with open(eval_output_file, "r", encoding="utf-8") as f:
        result = f.read()
        # load the json file
        try:
            result_json = json.loads(result)
        except json.JSONDecodeError:
            pytest.fail("Failed to parse workflow_output.json as valid JSON")

    assert result_json, f"The {eval_output_file} file is empty"
    assert isinstance(result_json, dict), f"The {eval_output_file} file is not a dictionary"
    assert result_json.get("average_score", 0) >= score_min, \
        f"The 'average_accuracy' is less than {score_min}"


@pytest.mark.e2e
async def test_eval():
    """
    Run the swe-bench evaluator with the golden patches and validate
    1. the workflow_output and
    2. swe_bench evaliation output
    """
    # Get package dynamically
    package_name = inspect.getmodule(SweBenchWorkflowConfig).__package__
    config_file: Path = importlib.resources.files(package_name).joinpath("configs", "config_gold.yml").absolute()

    # Create the configuration object for running the evaluation, single rep using the eval config in config.yml
    # WIP: skip test if eval config is not present
    config = EvaluationRunConfig(
        config_file=config_file,
        dataset=None,
        result_json_path="$",
        skip_workflow=False,
        skip_completed_entries=False,
        endpoint=None,
        endpoint_timeout=300,
        reps=1,
    )
    # Run evaluation
    eval_runner = EvaluationRun(config=config)
    output = await eval_runner.run_and_evaluate()

    # Ensure the workflow was not interrupted
    assert not output.workflow_interrupted, "The workflow was interrupted"

    # Validate the workflow output
    assert output.workflow_output_file, "The workflow_output.json file was not created"
    validate_workflow_output(output.workflow_output_file)

    # Look for the swe_bench evaluation out
    swe_bench_output_file: Path | None = None
    for output_file in output.evaluator_output_files:
        output_file_str = str(output_file)
        if "swe_bench_output.json" in output_file_str:
            swe_bench_output_file = output_file

    # Verify the rag_accuracy_output.json file
    assert swe_bench_output_file, "The rag_accuracy_output.json file was not created"
    validate_evaluation_output(swe_bench_output_file)
