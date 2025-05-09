import pytest
from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate

from aiq.builder.framework_enum import LLMFrameworkEnum
from aiq.builder.workflow_builder import WorkflowBuilder
from aiq.llm.aws_bedrock_llm import AWSBedrockModelConfig
from aiq.llm.nim_llm import NIMModelConfig
from aiq.llm.openai_llm import OpenAIModelConfig


@pytest.mark.integration
@pytest.mark.asyncio
async def test_nim_langchain_agent():
    """Test NIM LLM with LangChain agent. Requires NVIDIA_API_KEY to be set."""

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
@pytest.mark.asyncio
async def test_openai_langchain_agent():
    """Test OpenAI LLM with LangChain agent. Requires OPENAI_API_KEY to be set."""
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
@pytest.mark.asyncio
async def test_aws_bedrock_langchain_agent():
    """Test AWS Bedrock LLM with LangChain agent. Requires AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY to be set."""
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
