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

import asyncio
import logging
import sys
import time
from pathlib import Path

import pytest

logger = logging.getLogger(__name__)


@pytest.mark.integration
@pytest.mark.usefixtures("nvidia_api_key")
async def test_hitl_workflow(capsys):
    import nat_por_to_jiratickets.register  # noqa: F401
    from nat.runtime.loader import load_workflow
    from nat.test.utils import locate_example_config
    from nat_simple_calculator_hitl.retry_react_agent import RetryReactAgentConfig

    expected_prompt = "Please confirm if you would like to proceed"
    config_file: Path = locate_example_config(RetryReactAgentConfig, "config-hitl.yml")

    result = None
    async with load_workflow(config_file) as workflow:

        async with workflow.run("Is 2 * 4 greater than 5?") as runner:

            runner_future = asyncio.create_task(runner.result(to_type=str))
            deadline = time.time() + 120  # 2 minute timeout
            done = False
            prompted = False
            while not done and time.time() < deadline:
                captured = capsys.readouterr()
                if not prompted:
                    assert not runner_future.done(), "Runner finished before prompt detected"
                    if expected_prompt in captured.out:
                        prompted = True
                        sys.stdin.write("no\n")
                    else:
                        await asyncio.sleep(0.1)
                else:
                    done = runner_future.done()
                    if done:
                        assert runner_future.exception() is None, f"Runner failed with {runner_future.exception()}"
                        result = runner_future.result()
                    else:
                        await asyncio.sleep(0.1)

            if not done:
                runner_future.cancel()

    assert result is not None, "Test did not complete successfully"
    assert "I seem to be having a problem." in result.lower()
