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

import logging

from pydantic import Field

from nat.builder.builder import Builder
from nat.builder.framework_enum import LLMFrameworkEnum
from nat.builder.function_info import FunctionInfo
from nat.cli.register_workflow import register_function
from nat.data_models.function import FunctionBaseConfig

logger = logging.getLogger(__name__)


class ReactBenchmarkAgentFunctionConfig(FunctionBaseConfig, name="react_benchmark_agent"):
    """
    React Benchmark Agent for Agent Leaderboard evaluation.

    This function supports two modes:
    1. Standard mode: Acts as a regular tool in the workflow
    2. Decision-only mode: Dynamically registers tool stubs from dataset to capture tool intents
    """

    prefix: str = Field(default="Agent:", description="Prefix to add before responses.")
    decision_only: bool = Field(
        default=False,
        description="If True, register tool stubs from dataset to capture tool intents without execution.",
    )
    canned_response_template: str = Field(
        default="Successfully executed {tool_name}. Operation completed.",
        description="Template for canned responses in decision-only mode.",
    )


@register_function(config_type=ReactBenchmarkAgentFunctionConfig, framework_wrappers=[LLMFrameworkEnum.LANGCHAIN])
async def react_benchmark_agent_function(config: ReactBenchmarkAgentFunctionConfig, builder: Builder):
    """
    Registers the React Benchmark Agent function.

    In decision-only mode, this function initializes a tool intent buffer that can be used
    to dynamically register tool stubs from dataset tool schemas.

    Args:
        config (ReactBenchmarkAgentFunctionConfig): The configuration for the function.
        builder (Builder): The builder object.

    Returns:
        FunctionInfo: The function info object for the function.
    """
    # Import tool intent stub system if in decision-only mode
    if config.decision_only:
        from .tool_intent_stubs import ToolIntentBuffer

        # Create shared intent buffer
        intent_buffer = ToolIntentBuffer()

        # Store in builder runtime metadata for access by workflow and evaluators
        if not hasattr(builder, "runtime_metadata"):
            builder.runtime_metadata = {}
        builder.runtime_metadata["tool_intent_buffer"] = intent_buffer

        logger.info("Initialized tool intent buffer for decision-only mode")

        # In decision-only mode, this function just returns a status message
        async def _decision_only_info(query: str) -> str:
            """
            Provides information about the decision-only evaluation mode.

            Args:
                query (str): User query

            Returns:
                str: Information about decision-only mode
            """
            num_intents = len(intent_buffer.get_intents())
            return f"{config.prefix} Decision-only mode active. Captured {num_intents} tool intents so far."

        yield FunctionInfo.from_fn(_decision_only_info, description=_decision_only_info.__doc__)

    else:
        # Standard mode: echo function for testing
        async def _echo(text: str) -> str:
            """
            Takes a text input and echoes back with a pre-defined prefix.

            Args:
                text (str): The text to echo back.

            Returns:
                str: The text with the prefix.
            """
            return f"{config.prefix} {text}"

        yield FunctionInfo.from_fn(_echo, description=_echo.__doc__)
