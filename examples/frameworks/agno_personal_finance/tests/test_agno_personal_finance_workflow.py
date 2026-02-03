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


@pytest.mark.integration
@pytest.mark.usefixtures("serp_api_key", "openai_api_key")
async def test_full_workflow():
    from nat.test.utils import locate_example_config
    from nat.test.utils import run_workflow
    from nat_agno_personal_finance.agno_personal_finance_function import AgnoPersonalFinanceFunctionConfig

    config_file: Path = locate_example_config(AgnoPersonalFinanceFunctionConfig)

    await run_workflow(config_file=config_file,
                       question=("My financial goal is to retire at age 50. "
                                 "I am currently 30 years old, working as a Solutions Architect at NVIDIA."),
                       expected_answer="financial plan")
