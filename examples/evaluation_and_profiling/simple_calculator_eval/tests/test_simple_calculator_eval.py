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

import logging
from pathlib import Path

import pytest

from nat.eval.evaluate import EvaluationRun
from nat.eval.evaluate import EvaluationRunConfig
from nat.test.utils import locate_example_config
from nat.test.utils import validate_workflow_output

logger = logging.getLogger(__name__)


@pytest.mark.integration
@pytest.mark.usefixtures("nvidia_api_key")
async def test_eval():
    import nat_simple_calculator_eval

    # Get config dynamically
    config_file: Path = locate_example_config(nat_simple_calculator_eval, "config-tunable-rag-eval.yml")

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

    # Ensure the workflow was not interrupted
    assert not output.workflow_interrupted, "The workflow was interrupted"

    # Look for the tuneable_eval_output file
    tuneable_eval_output: Path | None = None

    for output_file in output.evaluator_output_files:
        assert output_file.exists()
        output_file_str = str(output_file)
        if "tuneable_eval_output" in output_file_str:
            tuneable_eval_output = output_file

    # Validate the workflow output
    assert output.workflow_output_file, "The workflow_output.json file was not created"
    validate_workflow_output(output.workflow_output_file)

    # Verify that at least one tuneable_eval_output file is present
    assert tuneable_eval_output, "Expected output file does not exist"
