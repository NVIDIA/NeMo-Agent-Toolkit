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

import json
import logging
from pathlib import Path

import pytest

from nat.eval.evaluate import EvaluationRun
from nat.eval.evaluate import EvaluationRunConfig
from nat.test.utils import locate_example_config
from nat.test.utils import validate_workflow_output

logger = logging.getLogger(__name__)


def validate_rag_accuracy(rag_metric_output_file: Path, score: float):
    """
    1. Validate the contents of the rag evaluator ouput file.
    2. Ensure the average_score is above a minimum threshold.
    WIP: output format should be published as a schema and this validation should be done against that schema.
    """
    # Ensure the ile exists
    assert rag_metric_output_file and rag_metric_output_file.exists(), \
        f"The {rag_metric_output_file} was not created"
    with open(rag_metric_output_file, encoding="utf-8") as f:
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
    with open(trajectory_output_file, encoding="utf-8") as f:
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


@pytest.mark.integration
@pytest.mark.usefixtures("nvidia_api_key")
async def test_eval():
    """
    1. nat-eval writes the workflow output to workflow_output.json
    2. nat-eval creates a file with scores for each evaluation metric.
    3. This test audits -
       a. the rag accuracy metric
       b. the trajectory score (if present)
    """
    import nat_simple_web_query_eval

    # Get config dynamically
    config_file: Path = locate_example_config(nat_simple_web_query_eval, "eval_config.yml")

    # Create the configuration object for running the evaluation, single rep using the eval config in eval_config.yml
    # WIP: skip test if eval config is not present
    config = EvaluationRunConfig(
        config_file=config_file,
        dataset=None,
        result_json_path="$",
        skip_workflow=False,
        skip_completed_entries=False,
        endpoint=None,
        endpoint_timeout=30,
        reps=1,
        override=(('eval.general.max_concurrency', '1'), ),
    )

    # Run evaluation
    eval_runner = EvaluationRun(config=config)
    output = await eval_runner.run_and_evaluate()

    assert eval_runner.eval_config is not None, "The eval config is not present"

    type_name_map = {}
    for eval_type in ["ragas", "trajectory"]:
        expected = []
        for name, config in eval_runner.eval_config.evaluators.items():
            if config.type == eval_type:
                expected.append(f"{name}_output.json")
        type_name_map[eval_type] = expected

    # Ensure the workflow was not interrupted
    assert not output.workflow_interrupted, "The workflow was interrupted"

    # Validate the workflow output
    assert output.workflow_output_file, "The workflow_output.json file was not created"
    validate_workflow_output(output.workflow_output_file)

    for output_file in output.evaluator_output_files:
        base_name = output_file.name
        if base_name in type_name_map["ragas"]:
            # Relevance and Groundedness should evaluate better than Accuracy
            min_score = 0.5 if "accuracy" in str(output_file) else 0.75
            validate_rag_accuracy(output_file, min_score)
        elif base_name in type_name_map["trajectory"]:
            validate_trajectory_accuracy(output_file)
