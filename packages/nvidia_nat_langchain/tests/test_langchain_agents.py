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
import os
from typing import Any

import pytest
from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate

from nat.builder.framework_enum import LLMFrameworkEnum
from nat.builder.workflow_builder import WorkflowBuilder
from nat.llm.aws_bedrock_llm import AWSBedrockModelConfig
from nat.llm.azure_openai_llm import AzureOpenAIModelConfig
from nat.llm.nim_llm import NIMModelConfig
from nat.llm.openai_llm import OpenAIModelConfig

try:
    from nat.llm.huggingface_llm import HuggingFaceConfig  # noqa: F401
    HAS_HUGGINGFACE = True
except ImportError:
    HAS_HUGGINGFACE = False


@pytest.mark.integration
@pytest.mark.usefixtures("nvidia_api_key")
async def test_nim_langchain_agent():
    """
    Test NIM LLM with LangChain/LangGraph agent. Requires NVIDIA_API_KEY to be set.
    """

    prompt = ChatPromptTemplate.from_messages([("system", "You are a helpful AI assistant."), ("human", "{input}")])

    llm_config = NIMModelConfig(model_name="meta/llama-3.1-70b-instruct", temperature=0.0)

    async with WorkflowBuilder() as builder:
        await builder.add_llm("nim_llm", llm_config)
        llm = await builder.get_llm("nim_llm", wrapper_type=LLMFrameworkEnum.LANGCHAIN)

        agent = prompt | llm

        response = await agent.ainvoke({"input": "What is 1+2?"})
        assert isinstance(response, AIMessage)
        assert response.content is not None
        assert isinstance(response.content, str)
        assert "3" in response.content.lower()


@pytest.mark.integration
@pytest.mark.usefixtures("openai_api_key")
async def test_openai_langchain_agent():
    """
    Test OpenAI LLM with LangChain/LangGraph agent. Requires OPENAI_API_KEY to be set.
    """
    prompt = ChatPromptTemplate.from_messages([("system", "You are a helpful AI assistant."), ("human", "{input}")])

    llm_config = OpenAIModelConfig(model_name="gpt-3.5-turbo", temperature=0.0)

    async with WorkflowBuilder() as builder:
        await builder.add_llm("openai_llm", llm_config)
        llm = await builder.get_llm("openai_llm", wrapper_type=LLMFrameworkEnum.LANGCHAIN)

        agent = prompt | llm

        response = await agent.ainvoke({"input": "What is 1+2?"})
        assert isinstance(response, AIMessage)
        assert response.content is not None
        assert isinstance(response.content, str)
        assert "3" in response.content.lower()


@pytest.mark.integration
@pytest.mark.usefixtures("aws_keys")
async def test_aws_bedrock_langchain_agent():
    """
    Test AWS Bedrock LLM with LangChain/LangGraph agent.
    Requires AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY to be set.
    See https://docs.aws.amazon.com/bedrock/latest/userguide/setting-up.html for more information.
    """
    prompt = ChatPromptTemplate.from_messages([("system", "You are a helpful AI assistant."), ("human", "{input}")])

    llm_config = AWSBedrockModelConfig(model_name="meta.llama3-3-70b-instruct-v1:0",
                                       temperature=0.0,
                                       region_name="us-east-2",
                                       max_tokens=1024)

    async with WorkflowBuilder() as builder:
        await builder.add_llm("aws_bedrock_llm", llm_config)
        llm = await builder.get_llm("aws_bedrock_llm", wrapper_type=LLMFrameworkEnum.LANGCHAIN)

        agent = prompt | llm

        response = await agent.ainvoke({"input": "What is 1+2?"})
        assert isinstance(response, AIMessage)
        assert response.content is not None
        assert isinstance(response.content, str)
        assert "3" in response.content.lower()


@pytest.mark.integration
@pytest.mark.usefixtures("azure_openai_keys")
@pytest.mark.parametrize("api_version", [None, '2025-04-01-preview'])
async def test_azure_openai_langchain_agent(api_version: str | None):
    """
    Test Azure OpenAI LLM with LangChain/LangGraph agent.
    Requires AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT to be set.
    The model can be changed by setting AZURE_OPENAI_DEPLOYMENT.
    See https://learn.microsoft.com/en-us/azure/ai-foundry/openai/quickstart for more information.
    """
    prompt = ChatPromptTemplate.from_messages([("system", "You are a helpful AI assistant."), ("human", "{input}")])

    config_args: dict[str, Any] = {"azure_deployment": os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt-4.1")}
    if api_version is not None:
        config_args["api_version"] = api_version
    llm_config = AzureOpenAIModelConfig(**config_args)

    async with WorkflowBuilder() as builder:
        await builder.add_llm("azure_openai_llm", llm_config)
        llm = await builder.get_llm("azure_openai_llm", wrapper_type=LLMFrameworkEnum.LANGCHAIN)

        agent = prompt | llm

        response = await agent.ainvoke({"input": "What is 1+2?"})
        assert isinstance(response, AIMessage)
        assert response.content is not None
        assert isinstance(response.content, str)
        assert "3" in response.content.lower()


@pytest.mark.integration
@pytest.mark.usefixtures("azure_openai_keys")
async def test_azure_openai_react_e2e(test_data_dir: str):
    from nat.test.utils import run_workflow

    config_file = os.path.join(test_data_dir, "azure_openai_e2e.yaml")
    await run_workflow(config_file=config_file, question="What is 1+2?", expected_answer="3")


@pytest.mark.integration
@pytest.mark.skipif(not HAS_HUGGINGFACE, reason="HuggingFace dependencies (transformers, torch) not installed")
async def test_huggingface_langchain_agent():
    """
    Test HuggingFace LLM with LangChain/LangGraph agent.
    Requires transformers and torch to be installed (optional dependencies).
    """
    from nat.llm.huggingface_llm import HuggingFaceConfig

    prompt = ChatPromptTemplate.from_messages([("system", "You are a helpful AI assistant."), ("human", "{input}")])

    # Use a small, fast model for testing
    llm_config = HuggingFaceConfig(model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0", temperature=0.0, max_new_tokens=50)

    async with WorkflowBuilder() as builder:
        await builder.add_llm("huggingface_llm", llm_config)
        llm = await builder.get_llm("huggingface_llm", wrapper_type=LLMFrameworkEnum.LANGCHAIN)

        prompt_result = await prompt.ainvoke({"input": "What is 1+2?"})
        response = await llm.ainvoke(prompt_result.to_messages())

        assert isinstance(response, AIMessage)
        assert response.content is not None
        assert isinstance(response.content, str)
        assert "3" in response.content.lower()
