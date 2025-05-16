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
import logging
from pathlib import Path

import pandas as pd
import pytest
import yaml
from aiq_alert_triage_agent.register import AlertTriageAgentWorkflowConfig

from aiq.runtime.loader import load_workflow

logger = logging.getLogger(__name__)


@pytest.mark.e2e
async def test_full_workflow():

    package_name = inspect.getmodule(AlertTriageAgentWorkflowConfig).__package__

    config_file: Path = importlib.resources.files(package_name).joinpath("configs", "config_test_mode.yml").absolute()

    with open(config_file, "r") as file:
        config = yaml.safe_load(file)
        output_filepath = config["workflow"]["test_output_path"]
    output_filepath_abs = importlib.resources.files(package_name).joinpath("../../../../", output_filepath).absolute()

    input_message = "run in test mode"
    async with load_workflow(config_file) as workflow:

        async with workflow.run(input_message) as runner:

            result = await runner.result(to_type=str)

        assert result == f"Successfully processed 4 alerts. Results saved to {output_filepath}"

        output_df = pd.read_csv(output_filepath_abs)

        # Check that the output dataframe has the correct number of rows and columns
        assert output_df.shape[0] == 4
        assert output_df.shape[1] == 11
        assert output_df.columns[-1] == "output"

        # Check that all rows in 'output' column contain non-empty strings
        assert all(isinstance(output, str) and len(output.strip()) > 0 for output in output_df["output"])

        # Deterministic data point: host under maintenance
        assert 'maintenance' in output_df.iloc[3]["output"]

        # Check that rows 0-2 (hosts not under maintenance) contain root cause categorization
        for i in range(3):
            assert "root cause category" in output_df.iloc[i]["output"].lower()
