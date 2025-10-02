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
import yaml

from nat.runtime.loader import load_workflow
from nat.test.utils import locate_example_config
from nat_alert_triage_agent.register import AlertTriageAgentWorkflowConfig

logger = logging.getLogger(__name__)


@pytest.mark.integration
@pytest.mark.usefixtures("nvidia_api_key")
async def test_full_workflow(root_repo_dir: Path):

    config_file: Path = locate_example_config(AlertTriageAgentWorkflowConfig, "config_offline_mode.yml")

    with open(config_file, encoding="utf-8") as file:
        config = yaml.safe_load(file)
        input_filepath = config["eval"]["general"]["dataset"]["file_path"]

    input_filepath_abs = root_repo_dir.joinpath(input_filepath).absolute()

    assert input_filepath_abs.exists(), f"Input data file {input_filepath_abs} does not exist"

    # Load input data
    with open(input_filepath_abs, encoding="utf-8") as f:
        input_data = json.load(f)

    input_data = input_data[0]  # Limit to first row for testing

    # Run the workflow
    async with load_workflow(config_file) as workflow:
        async with workflow.run(input_data["question"]) as runner:
            result = await runner.result(to_type=str)

    # Check that the results are as expected
    assert len(result) > 0, "Result is empty"

    # Deterministic data point: host under maintenance
    assert input_data['label'] in result

    # Check that rows with hosts not under maintenance contain root cause categorization
    assert "root cause category" in result.lower()
