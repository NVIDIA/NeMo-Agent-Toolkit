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

from nat.builder.builder import Builder
from nat.builder.function_info import FunctionInfo
from nat.cli.register_workflow import register_function
from nat.data_models.component_ref import LLMRef
from nat.data_models.function import FunctionBaseConfig

logger = logging.getLogger(__name__)


class HaystackChitchatConfig(FunctionBaseConfig, name="haystack_chitchat_agent"):
    llm_name: LLMRef


@register_function(config_type=HaystackChitchatConfig)
async def haystack_chitchat_agent_as_tool(tool_config: HaystackChitchatConfig, builder: Builder):

    from haystack_integrations.components.generators.nvidia import NvidiaGenerator

    generator = NvidiaGenerator(
        model=tool_config.llm_name,
        api_url="https://integrate.api.nvidia.com/v1",
        model_arguments={
            "temperature": 0.2,
            "top_p": 0.7,
            "max_tokens": 1024,
        },
    )

    generator.warm_up()

    async def _arun(inputs: str) -> str:
        """
        using web search on a given topic extracted from user input
        Args:
            inputs : user input
        """
        out = generator.run(prompt=inputs)
        output = out["replies"][0]  # noqa: W293 E501

        logger.info("output from langchain_research_tool: %s", output)  # noqa: W293 E501
        return output

    yield FunctionInfo.from_fn(_arun, description="extract relevent information from search the web")  # noqa: W293 E501
