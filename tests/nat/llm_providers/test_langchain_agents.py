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

import pytest
from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate

from nat.builder.framework_enum import LLMFrameworkEnum
from nat.builder.workflow_builder import WorkflowBuilder
from nat.llm.aws_bedrock_llm import AWSBedrockModelConfig
from nat.llm.azure_openai_llm import AzureOpenAIModelConfig
from nat.llm.nim_llm import NIMModelConfig
from nat.llm.openai_llm import OpenAIModelConfig


@pytest.mark.integration
async def test_nim_langchain_agent():
    """
    Test NIM LLM with LangChain agent. Requires NVIDIA_API_KEY to be set.
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
async def test_openai_langchain_agent():
    """
    Test OpenAI LLM with LangChain agent. Requires OPENAI_API_KEY to be set.
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
async def test_aws_bedrock_langchain_agent():
    """
    Test AWS Bedrock LLM with LangChain agent.
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
async def test_azure_openai_langchain_agent():
    """
    Test Azure OpenAI LLM with LangChain agent.
    Requires AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT to be set.
    The model can be changed by setting AZURE_OPENAI_DEPLOYMENT.
    See https://learn.microsoft.com/en-us/azure/ai-foundry/openai/quickstart for more information.
    """
    prompt = ChatPromptTemplate.from_messages([("system", "You are a helpful AI assistant."), ("human", "{input}")])

    llm_config = AzureOpenAIModelConfig(azure_deployment=os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt-4.1"))

    async with WorkflowBuilder() as builder:
        await builder.add_llm("azure_openai_llm", llm_config)
        llm = await builder.get_llm("azure_openai_llm", wrapper_type=LLMFrameworkEnum.LANGCHAIN)

        agent = prompt | llm

        response = await agent.ainvoke({"input": "What is 1+2?"})
        assert isinstance(response, AIMessage)
        assert response.content is not None
        assert isinstance(response.content, str)
        assert "3" in response.content.lower()
