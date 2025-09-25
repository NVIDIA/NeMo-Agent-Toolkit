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
from collections.abc import AsyncGenerator

from pydantic import Field

from nat.builder.builder import Builder
from nat.builder.framework_enum import LLMFrameworkEnum
from nat.builder.function_info import FunctionInfo
from nat.cli.register_workflow import register_function
from nat.data_models.function import FunctionBaseConfig

from . import nat_time_mcp_tool  # noqa: F401 # pylint: disable=W0611     #imported for side effects (registration)
from . import weather_update_tool  # noqa: F401 # pylint: disable=W0611     #imported for side effects (registration)

logger = logging.getLogger(__name__)


class ADKFunctionConfig(FunctionBaseConfig, name="adk"):
    """Configuration for ADK demo function."""
    description: str
    llm_name: str
    tool_names: list[str] = Field(default_factory=list)
    workflow_alias: str = Field(default="adk_function")


@register_function(config_type=ADKFunctionConfig, framework_wrappers=[LLMFrameworkEnum.ADK])
async def adk_demo(
    config: ADKFunctionConfig,
    _builder: Builder,
) -> AsyncGenerator[FunctionInfo, None]:
    """An example function that demonstrates how to use the Google ADK framework with NAT.

    Args:
        config (ADKFunctionConfig): The configuration for the ADK demo function.
        _builder (Builder): The NAT builder instance.
    Yields:
        AsyncGenerator[FunctionInfo, None]: An async generator that yields a FunctionInfo object.
    """

    from google.adk import Runner
    from google.adk.agents import Agent
    from google.adk.artifacts import InMemoryArtifactService
    from google.adk.sessions import InMemorySessionService
    from google.genai import types

    if not config.llm_name:
        raise ValueError("llm_name must be provided")
    if not config.tool_names:
        raise ValueError("tool_names must contain at least one tool")

    model = await _builder.get_llm(config.llm_name, wrapper_type=LLMFrameworkEnum.ADK)
    tools = await _builder.get_tools(config.tool_names, wrapper_type=LLMFrameworkEnum.ADK)

    agent = Agent(
        name="weather_time_agent",
        model=model,
        description=("Agent to answer questions about the time and weather in a city."),
        instruction=("You are a helpful agent who can answer user questions "
                     "about weather in a city. You also have a sub-agent "
                     "that can answer questions about the time in a city."),
        tools=tools,
    )

    # Initialize the Runner with the agent and services
    app_name = "my_app"
    session_service = InMemorySessionService()
    artifact_service = InMemoryArtifactService()
    runner = Runner(
        app_name=app_name,
        agent=agent,
        artifact_service=artifact_service,
        session_service=session_service,
    )
    user_id = "user1"
    session = await session_service.create_session(app_name=app_name, user_id=user_id)

    async def _response_fn(input_message: str) -> str:
        """Wrapper for response fn

        Args:
            input_message (str): The input message from the user.
        Returns:
            str : The response from the agent.
        """

        async def run_prompt(new_message: str) -> str:
            """Run prompt through the agent.

            Args:
                new_message (str): The input message from the user.
            Returns:
                str: The response from the agent.
            """
            content = types.Content(role="user", parts=[types.Part.from_text(text=new_message)])
            # print("** User says:", content.model_dump(exclude_none=True))
            text_buf: list[str] = []
            async for event in runner.run_async(
                    user_id=user_id,
                    session_id=session.id,
                    new_message=content,
            ):
                parts = getattr(getattr(event, "content", None), "parts", None)
                if parts:
                    for part in parts:
                        if getattr(part, "text", None):
                            text_buf.append(part.text)
            return "".join(text_buf) if text_buf else ""

        return await run_prompt(input_message)

    def convert_dict_to_str(response: dict) -> str:
        """Convert a dict object to str (takes the 'output' field).

        Args:
            response (dict): The response from the agent.

        Returns:
            str: The response as a string."""
        return response["output"]

    try:
        yield FunctionInfo.create(single_fn=_response_fn)
    except GeneratorExit:
        logger.debug("Exited early!", exc_info=True)
    finally:
        logger.debug("Cleaning up")
