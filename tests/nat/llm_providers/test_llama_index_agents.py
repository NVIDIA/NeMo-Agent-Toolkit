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
import os
from typing import Any

import pytest
from llama_index.core.agent import ReActAgent
from llama_index.core.tools import BaseTool
from llama_index.core.tools import FunctionTool

from nat.builder.framework_enum import LLMFrameworkEnum
from nat.builder.workflow_builder import WorkflowBuilder
from nat.llm.aws_bedrock_llm import AWSBedrockModelConfig
from nat.llm.azure_openai_llm import AzureOpenAIModelConfig
from nat.llm.nim_llm import NIMModelConfig
from nat.llm.openai_llm import OpenAIModelConfig


def calculator(expression: str) -> str:
    """Calculate the result of a mathematical expression.

    Args:
        expression: A string containing a mathematical expression (e.g., "2 + 2")

    Returns:
        The result of the calculation as a string
    """
    try:
        # Safely evaluate the expression
        result = eval(expression)  # pylint: disable=eval-used
        return str(result)
    except Exception as e:
        return f"Error calculating expression: {str(e)}"


async def create_minimal_agent(llm_name: str, llm_config: Any) -> ReActAgent:
    """Helper function to create a minimal agent with the specified LLM."""
    async with WorkflowBuilder() as builder:
        await builder.add_llm(llm_name, llm_config)
        llm = await builder.get_llm(llm_name, wrapper_type=LLMFrameworkEnum.LLAMA_INDEX)

        tools: list[BaseTool] = [
            FunctionTool.from_defaults(fn=calculator,
                                       name="tool",
                                       description="Use this tool to perform mathematical calculations. "
                                       "Input should be a string containing a mathematical expression.")
        ]

        return ReActAgent.from_tools(tools=tools, llm=llm, verbose=True)


@pytest.mark.integration
async def test_nim_minimal_agent():
    """Test NIM LLM with minimal LlamaIndex agent. Requires NVIDIA_API_KEY to be set."""
    llm_config = NIMModelConfig(model_name="meta/llama-3.1-70b-instruct", temperature=0.0)
    agent = await create_minimal_agent("nim_llm", llm_config)

    response = await agent.achat("What is 1+2?")
    assert response is not None
    assert hasattr(response, 'response')
    assert "3" in response.response.lower()


@pytest.mark.integration
async def test_openai_minimal_agent():
    """Test OpenAI LLM with minimal LlamaIndex agent. Requires OPENAI_API_KEY to be set."""
    llm_config = OpenAIModelConfig(model_name="gpt-3.5-turbo", temperature=0.0)
    agent = await create_minimal_agent("openai_llm", llm_config)

    response = await agent.achat("What is 1+2?")
    assert response is not None
    assert hasattr(response, 'response')
    assert "3" in response.response.lower()


@pytest.mark.integration
async def test_aws_bedrock_minimal_agent():
    """
    Test AWS Bedrock LLM with LangChain agent.
    Requires AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY to be set.
    See https://docs.aws.amazon.com/bedrock/latest/userguide/setting-up.html for more information.
    """
    llm_config = AWSBedrockModelConfig(model_name="us.meta.llama3-1-405b-instruct-v1:0",
                                       temperature=0.0,
                                       region_name="us-east-2",
                                       context_size=1024,
                                       credentials_profile_name="default")
    agent = await create_minimal_agent("aws_bedrock_llm", llm_config)

    response = await agent.achat("What is 1+2?")
    assert response is not None
    assert hasattr(response, 'response')
    assert "3" in response.response.lower()


@pytest.mark.integration
async def test_azure_openai_minimal_agent():
    """
    Test Azure OpenAI LLM with minimal LlamaIndex agent.
    Requires AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT to be set.
    The model can be changed by setting AZURE_OPENAI_DEPLOYMENT.
    See https://learn.microsoft.com/en-us/azure/ai-foundry/openai/quickstart for more information.
    """
    llm_config = AzureOpenAIModelConfig(azure_deployment=os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt-4.1"))
    agent = await create_minimal_agent("azure_openai_llm", llm_config)

    response = await agent.achat("What is 1+2?")
    assert response is not None
    assert hasattr(response, 'response')
    assert "3" in response.response.lower()
