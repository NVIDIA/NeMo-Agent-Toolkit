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

from nat.builder.framework_enum import LLMFrameworkEnum


class MockBuilder:
    """Mock builder that validates Agno workflow component lookups."""

    def __init__(self, expected_llm_name: str, expected_tool_names: list[str]) -> None:
        """Initialize expected component lookup values.

        Args:
            expected_llm_name: LLM component name the workflow should request.
            expected_tool_names: Tool component names the workflow should request.
        """
        self.expected_llm_name = expected_llm_name
        self.expected_tool_names = expected_tool_names
        self.llm_requested = False
        self.tools_requested = False

    async def get_llm(self, llm_name: str, wrapper_type: LLMFrameworkEnum) -> None:
        """Validate that the workflow requests the expected Agno LLM.

        Args:
            llm_name: LLM component name requested by the workflow.
            wrapper_type: Framework wrapper requested by the workflow.

        Returns:
            None: The test does not execute the LLM.
        """
        assert llm_name == self.expected_llm_name
        assert wrapper_type == LLMFrameworkEnum.AGNO
        self.llm_requested = True

    async def get_tools(self, tool_names: list[str], wrapper_type: LLMFrameworkEnum) -> list[object]:
        """Validate that the workflow requests the expected Agno tools.

        Args:
            tool_names: Tool component names requested by the workflow.
            wrapper_type: Framework wrapper requested by the workflow.

        Returns:
            Empty list of mock tools.
        """
        assert tool_names == self.expected_tool_names
        assert wrapper_type == LLMFrameworkEnum.AGNO
        self.tools_requested = True

        return []


async def test_workflow_initializes_with_agno_v2_constructor_args():
    from nat.builder.function_info import FunctionInfo
    from nat_agno_personal_finance.agno_personal_finance_function import AgnoPersonalFinanceFunctionConfig
    from nat_agno_personal_finance.agno_personal_finance_function import agno_personal_finance_function

    config = AgnoPersonalFinanceFunctionConfig(llm_name="mock_llm", tools=["mock_tool"])
    builder = MockBuilder(expected_llm_name="mock_llm", expected_tool_names=["mock_tool"])

    async with agno_personal_finance_function(config, builder) as fn_info:
        assert isinstance(fn_info, FunctionInfo)
        assert builder.llm_requested
        assert builder.tools_requested


@pytest.mark.integration
@pytest.mark.usefixtures("serp_api_key", "nvidia_api_key")
async def test_full_workflow():
    from nat.test.utils import locate_example_config
    from nat.test.utils import run_workflow
    from nat_agno_personal_finance.agno_personal_finance_function import AgnoPersonalFinanceFunctionConfig

    config_file: Path = locate_example_config(AgnoPersonalFinanceFunctionConfig)

    await run_workflow(config_file=config_file,
                       question=("My financial goal is to retire at age 50. "
                                 "I am currently 30 years old, working as a Solutions Architect at NVIDIA."),
                       expected_answer="financial plan")
