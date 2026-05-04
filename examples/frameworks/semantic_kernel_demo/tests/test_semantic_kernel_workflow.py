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

from pathlib import Path

import pytest


@pytest.mark.usefixtures("mem0_api_key", "openai_api_key")
@pytest.mark.integration
async def test_full_workflow():
    from nat.test.utils import locate_example_config
    from nat.test.utils import run_workflow
    from nat_semantic_kernel_demo.register import SKTravelPlanningWorkflowConfig

    config_file: Path = locate_example_config(SKTravelPlanningWorkflowConfig)

    await run_workflow(
        config_file=config_file,
        question=("Create a 3-day travel itinerary for Tokyo in April, covering hotels and activities within a USD "
                  "2000 budget."),
        expected_answer="budget")
