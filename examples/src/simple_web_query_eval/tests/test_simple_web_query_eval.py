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

import aiq_simple_web_query_eval
import pytest

from aiq.eval.evaluate import EvaluationRun
from aiq.eval.evaluate import EvaluationRunConfig

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
    required_keys = ["id", "question", "answer", "generated_answer", "intermediate_steps"]
    for key in required_keys:
        assert all(item.get(key) for item in result_json), f"The '{key}' key is missing in workflow_output.json"


def validate_rag_accuracy(rag_metric_output_file: Path, score: float):
    """
    1. Validate the contents of the rag evaluator ouput file.
    2. Ensure the average_score is above a minimum threshold.
    WIP: output format should be published as a schema and this validation should be done against that schema.
    """
    # Ensure the ile exists
    assert rag_metric_output_file and rag_metric_output_file.exists(), \
        f"The {rag_metric_output_file} was not created"
    with open(rag_metric_output_file, "r", encoding="utf-8") as f:
        result = f.read()
        # load the json file
        try:
            result_json = json.loads(result)
        except json.JSONDecodeError:
            pytest.fail("Failed to parse workflow_output.json as valid JSON")

    assert result_json, f"The {rag_metric_output_file} file is empty"
    assert isinstance(result_json, dict), f"The {rag_metric_output_file} file is not a dictionary"
    assert result_json.get("average_score", 0) > score, \
        f"The {rag_metric_output_file} score is less than {score}"


def validate_trajectory_accuracy(trajectory_output_file: Path):
    """
    1. Validate the contents of the trajectory_output.json file.
    2. Ensure the average_score is above a minimum threshold.
    WIP: output format should be published as a schema and this validation should be done against that schema.
    """

    # Ensure the trajectory_output.json file exists
    assert trajectory_output_file and trajectory_output_file.exists(), "The trajectory_output.json file was not created"

    trajectory_score_min = 0.1
    with open(trajectory_output_file, "r", encoding="utf-8") as f:
        result = f.read()
        # load the json file
        try:
            result_json = json.loads(result)
        except json.JSONDecodeError:
            pytest.fail("Failed to parse workflow_output.json as valid JSON")

    assert result_json, "The trajectory_output.json file is empty"
    assert isinstance(result_json, dict), "The trajectory_output.json file is not a dictionary"
    assert result_json.get("average_score", 0) > trajectory_score_min, \
        f"The 'average_score' is less than {trajectory_score_min}"


@pytest.mark.e2e
async def test_eval():
    """
    1. aiq-eval writes the workflow output to workflow_output.json
    2. aiq-eval creates a file with scores for each evaluation metric.
    3. This test audits -
       a. the rag accuracy metric
       b. the trajectory score (if present)
    """
    # Get package dynamically
    package_name = inspect.getmodule(aiq_simple_web_query_eval).__package__
    config_file: Path = importlib.resources.files(package_name).joinpath("configs", "eval_config.yml").absolute()

    # Create the configuration object for running the evaluation, single rep using the eval config in eval_config.yml
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

    # Look for the ragas evaluator and trajectory evaluator output files
    rag_output_files: list[Path] = []
    trajectory_output_file: Path | None = None

    for output_file in output.evaluator_output_files:
        output_file_str = str(output_file)
        if "rag_" in output_file_str:
            rag_output_files.append(output_file)
        if "trajectory_output.json" in output_file_str:
            trajectory_output_file = output_file

    # Validate the workflow output
    assert output.workflow_output_file, "The workflow_output.json file was not created"
    validate_workflow_output(output.workflow_output_file)

    # Verify that atleast one rag metric output file is present
    assert rag_output_files, "Atleast one rag metric output whould be present"
    for rag_output_file in rag_output_files:
        # Relevance and Groundedness should evaluate better than Accuracy
        min_score = 0.5 if "accuracy" in str(rag_output_file) else 0.75
        validate_rag_accuracy(rag_output_file, min_score)

    # Verify the trajectory_output.json file
    if trajectory_output_file:
        validate_trajectory_accuracy(trajectory_output_file)
