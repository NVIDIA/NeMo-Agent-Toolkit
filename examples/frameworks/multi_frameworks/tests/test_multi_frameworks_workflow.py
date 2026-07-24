# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import pytest


def test_multi_frameworks_router_routes_missing_worker_choice_to_supervisor():
    from nat_multi_frameworks.register import _route_multi_frameworks_state

    assert _route_multi_frameworks_state({"input": "hello"}) == "supervisor"


def test_multi_frameworks_router_routes_worker_choice_to_workers():
    from nat_multi_frameworks.register import _route_multi_frameworks_state

    assert _route_multi_frameworks_state({"input": "hello", "chosen_worker_agent": "General"}) == "workers"


def test_multi_frameworks_router_routes_final_output_to_end():
    from nat_multi_frameworks.register import _route_multi_frameworks_state

    state = {"input": "hello", "chosen_worker_agent": "General", "final_output": "done"}

    assert _route_multi_frameworks_state(state) == "end"


@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.usefixtures("nvidia_api_key")
async def test_full_workflow():
    from nat.test.utils import locate_example_config
    from nat.test.utils import run_workflow
    from nat_multi_frameworks.register import MultiFrameworksWorkflowConfig

    config_file = locate_example_config(MultiFrameworksWorkflowConfig)

    await run_workflow(config_file=config_file, question="tell me about this workflow", expected_answer="workflow")
