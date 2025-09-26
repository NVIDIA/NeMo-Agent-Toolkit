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

from pydantic import Field

from nat.builder.builder import Builder
from nat.builder.framework_enum import LLMFrameworkEnum
from nat.builder.function_info import FunctionInfo
from nat.cli.register_workflow import register_function
from nat.data_models.component_ref import LLMRef
from nat.data_models.function import FunctionBaseConfig

logger = logging.getLogger(__name__)


class ADKFunctionConfig(FunctionBaseConfig, name="adk"):
    """Configuration for ADK demo function."""
    name: str = Field(default="nat-adk-agent")
    description: str
    prompt: str
    llm: LLMRef
    tool_names: list[str] = Field(default_factory=list)
    workflow_alias: str = Field(default="adk_agent")
    user_id: str = Field(default="nat")


@register_function(config_type=ADKFunctionConfig, framework_wrappers=[LLMFrameworkEnum.ADK])
async def adk_agent(config: ADKFunctionConfig, builder: Builder):
    """An example function that demonstrates how to use the Google ADK framework with NAT.

    Args:
        config (ADKFunctionConfig): The configuration for the ADK agent function.
        builder (Builder): The NAT builder instance.
    """
    import logging

    from google.adk import Runner
    from google.adk.agents import Agent
    from google.adk.artifacts import InMemoryArtifactService
    from google.adk.sessions import InMemorySessionService
    from google.genai import types

    logging.getLogger("LiteLLM").setLevel(logging.WARNING)

    model = await builder.get_llm(config.llm, wrapper_type=LLMFrameworkEnum.ADK)
    tools = await builder.get_tools(config.tool_names, wrapper_type=LLMFrameworkEnum.ADK)

    agent = Agent(
        name=config.name,
        model=model,
        description=config.description,
        instruction=config.prompt,
        tools=tools,
    )

    # Initialize the Runner with the agent and services
    session_service = InMemorySessionService()
    artifact_service = InMemoryArtifactService()
    runner = Runner(app_name=config.name,
                    agent=agent,
                    artifact_service=artifact_service,
                    session_service=session_service)
    session = await session_service.create_session(app_name=config.name, user_id=config.user_id)

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
            text_buf: list[str] = []
            async for event in runner.run_async(user_id=config.user_id, session_id=session.id, new_message=content):
                if event.content is None:
                    continue
                if event.content.parts is None:
                    continue
                text_buf.extend(part.text for part in event.content.parts if part.text is not None)
            return "".join(text_buf) if text_buf else ""

        return await run_prompt(input_message)

    try:
        yield FunctionInfo.create(single_fn=_response_fn, description=config.description)
    except GeneratorExit:
        logger.debug("Exited early!", exc_info=True)
    finally:
        logger.debug("Cleaning up")
