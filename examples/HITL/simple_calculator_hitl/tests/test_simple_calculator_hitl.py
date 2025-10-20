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

import re
import subprocess
from pathlib import Path

import pytest


@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.usefixtures("nvidia_api_key")
@pytest.mark.parametrize("response, expected_result", [("no", "I seem to be having a problem"), ("yes", "Yes")],
                         ids=["no", "yes"])
def test_hitl_workflow(env_without_nat_log_level: dict[str, str], response: str, expected_result: str):
    from nat.test.utils import locate_example_config
    from nat_simple_calculator_hitl.retry_react_agent import RetryReactAgentConfig
    expected_prompt = "Please confirm if you would like to proceed"
    config_file: Path = locate_example_config(RetryReactAgentConfig, "config-hitl.yml")

    # Use subprocess to run the NAT CLI rather than using the API for two reasons:
    # 1) The HITL callback function requires a hook which is only available using the console front-end
    # 2) Pytest sets stdin to NULL by default
    # 3) The CI environment has NAT_LOG_LEVEL=WARNING which prevents the workflow result from being printed to stderr
    cmd = ["nat", "run", "--config_file", str(config_file.absolute()), "--input", '"Is 2 * 4 greater than 5?"']
    proc = subprocess.Popen(cmd,
                            stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT,
                            text=True,
                            env=env_without_nat_log_level)

    (stdout, _) = proc.communicate(input=f"{response}\n", timeout=60)
    assert proc.returncode == 0, f"Process failed with return code {proc.returncode}\noutput: {stdout}"
    assert expected_prompt in stdout

    result_pattern = re.compile(f"Workflow Result:.*{expected_result}", re.IGNORECASE | re.MULTILINE | re.DOTALL)
    assert result_pattern.search(stdout) is not None, \
        f"Expected result '{expected_result}' not found in output: {stdout}"
