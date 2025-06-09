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

import importlib
import importlib.resources
import inspect
import json
import logging
from pathlib import Path

import pytest
import yaml
from aiq_alert_triage_agent.register import AlertTriageAgentWorkflowConfig

from aiq.runtime.loader import load_workflow

logger = logging.getLogger(__name__)


@pytest.mark.e2e
async def test_full_workflow():

    package_name = inspect.getmodule(AlertTriageAgentWorkflowConfig).__package__

    config_file: Path = importlib.resources.files(package_name).joinpath("configs",
                                                                         "config_offline_mode.yml").absolute()

    with open(config_file, "r") as file:
        config = yaml.safe_load(file)
        input_filepath = config["eval"]["general"]["dataset"]["file_path"]

    input_filepath_abs = importlib.resources.files(package_name).joinpath("../../../../", input_filepath).absolute()

    # Load input data
    with open(input_filepath_abs, 'r') as f:
        input_data = json.load(f)

    # Run the workflow
    results = []
    async with load_workflow(config_file) as workflow:
        for item in input_data:
            async with workflow.run(item["question"]) as runner:
                result = await runner.result(to_type=str)
                results.append(result)

    # Check that the results are as expected
    assert len(results) == len(input_data)
    for i, result in enumerate(results):
        assert len(result) > 0, f"Result for item {i} is empty"

    # Deterministic data point: host under maintenance
    assert 'maintenance' in results[3]

    # Check that rows 0-2 (hosts not under maintenance) contain root cause categorization
    for i in range(3):
        assert "root cause category" in results[i].lower()
