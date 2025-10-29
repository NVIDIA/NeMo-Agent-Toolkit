# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
# pylint: disable=unused-argument

import logging
from collections.abc import Sequence
from typing import TypeVar

from nat.builder.builder import Builder
from nat.builder.framework_enum import LLMFrameworkEnum
from nat.cli.register_workflow import register_llm_client
from nat.data_models.llm import APITypeEnum
from nat.data_models.llm import LLMBaseConfig
from nat.data_models.retry_mixin import RetryMixin
from nat.data_models.thinking_mixin import ThinkingMixin
from nat.llm.aws_bedrock_llm import AWSBedrockModelConfig
from nat.llm.aws_sagemaker_llm import AWSSageMakerModelConfig
from nat.llm.azure_openai_llm import AzureOpenAIModelConfig
from nat.llm.litellm_llm import LiteLlmModelConfig
from nat.llm.nim_llm import NIMModelConfig
from nat.llm.openai_llm import OpenAIModelConfig
from nat.llm.utils.thinking import BaseThinkingInjector
from nat.llm.utils.thinking import FunctionArgumentWrapper
from nat.llm.utils.thinking import patch_with_thinking
from nat.utils.exception_handlers.automatic_retries import patch_with_retry
from nat.utils.responses_api import validate_no_responses_api
from nat.utils.type_utils import override

logger = logging.getLogger(__name__)

ModelType = TypeVar("ModelType")


def _patch_llm_based_on_config(client: ModelType, llm_config: LLMBaseConfig) -> ModelType:

    from langchain_core.language_models import LanguageModelInput
    from langchain_core.messages import BaseMessage
    from langchain_core.messages import HumanMessage
    from langchain_core.messages import SystemMessage
    from langchain_core.prompt_values import PromptValue

    class LangchainThinkingInjector(BaseThinkingInjector):

        @override
        def inject(self, messages: LanguageModelInput, *args, **kwargs) -> FunctionArgumentWrapper:
            """
            Inject a system prompt into the messages.

            The messages are the first (non-object) argument to the function.
            The rest of the arguments are passed through unchanged.

            Args:
                messages: The messages to inject the system prompt into.
                *args: The rest of the arguments to the function.
                **kwargs: The rest of the keyword arguments to the function.

            Returns:
                FunctionArgumentWrapper: An object that contains the transformed args and kwargs.

            Raises:
                ValueError: If the messages are not a valid type for LanguageModelInput.
            """
            if isinstance(messages, PromptValue):
                messages = messages.to_messages()
            elif isinstance(messages, str):
                messages = [HumanMessage(content=messages)]

            if isinstance(messages, Sequence) and all(isinstance(m, BaseMessage) for m in messages):
                for i, message in enumerate(messages):
                    if isinstance(message, SystemMessage):
                        if self.system_prompt not in str(message.content):
                            messages = list(messages)
                            messages[i] = SystemMessage(content=f"{message.content}\n{self.system_prompt}")
                        break
                else:
                    messages = list(messages)
                    messages.insert(0, SystemMessage(content=self.system_prompt))
                return FunctionArgumentWrapper(messages, *args, **kwargs)
            raise ValueError(f"Unsupported message type: {type(messages)}")

    if isinstance(llm_config, RetryMixin):
        client = patch_with_retry(client,
                                  retries=llm_config.num_retries,
                                  retry_codes=llm_config.retry_on_status_codes,
                                  retry_on_messages=llm_config.retry_on_errors)

    if isinstance(llm_config, ThinkingMixin) and llm_config.thinking_system_prompt is not None:
        client = patch_with_thinking(
            client,
            LangchainThinkingInjector(
                system_prompt=llm_config.thinking_system_prompt,
                function_names=[
                    "invoke",
                    "ainvoke",
                    "stream",
                    "astream",
                ],
            ))

    return client


@register_llm_client(config_type=AWSBedrockModelConfig, wrapper_type=LLMFrameworkEnum.LANGCHAIN)
async def aws_bedrock_langchain(llm_config: AWSBedrockModelConfig, _builder: Builder):

    from langchain_aws import ChatBedrockConverse

    validate_no_responses_api(llm_config, LLMFrameworkEnum.LANGCHAIN)

    client = ChatBedrockConverse(**llm_config.model_dump(
        exclude={"type", "context_size", "thinking", "api_type"},
        by_alias=True,
        exclude_none=True,
    ))

    yield _patch_llm_based_on_config(client, llm_config)


@register_llm_client(config_type=AWSSageMakerModelConfig, wrapper_type=LLMFrameworkEnum.LANGCHAIN)
async def aws_sagemaker_langchain(llm_config: AWSSageMakerModelConfig, _builder: Builder):

    from langchain_aws import SagemakerEndpoint
    from langchain_aws.llms.sagemaker_endpoint import LLMContentHandler
    from langchain_core.language_models import BaseChatModel
    from langchain_core.language_models import BaseLLM
    from langchain_core.messages import AIMessage
    from langchain_core.messages import BaseMessage
    from langchain_core.messages import HumanMessage
    from langchain_core.messages import SystemMessage
    from langchain_core.outputs import ChatGeneration
    from langchain_core.outputs import ChatResult

    validate_no_responses_api(llm_config, LLMFrameworkEnum.LANGCHAIN)

    # Create a default content handler if not provided
    class DefaultContentHandler(LLMContentHandler):
        content_type = "application/json"
        accepts = "application/json"

        def transform_input(self, prompt: str, model_kwargs: dict) -> bytes:
            import json
            # Format for Mistral models on SageMaker JumpStart
            # These models expect specific parameter names
            input_dict = {
                "inputs": prompt,
                "parameters": {
                    "max_new_tokens": model_kwargs.get("max_new_tokens", 1024),
                    "temperature": model_kwargs.get("temperature", 0.5),
                    "top_p": model_kwargs.get("top_p", 1.0),
                    "do_sample": True,
                }
            }
            return json.dumps(input_dict).encode('utf-8')

        def transform_output(self, output: bytes) -> str:
            import json
            # Handle StreamingBody from boto3
            if hasattr(output, 'read'):
                output_bytes = output.read()
            else:
                output_bytes = output

            response_json = json.loads(output_bytes.decode("utf-8"))

            # Handle different response formats from SageMaker
            # Format 1: List with generated_text
            if isinstance(response_json, list) and len(response_json) > 0:
                if isinstance(response_json[0], dict) and "generated_text" in response_json[0]:
                    return response_json[0]["generated_text"]
            # Format 2: Dict with generated_text
            elif isinstance(response_json, dict):
                if "generated_text" in response_json:
                    return response_json["generated_text"]
                # Format 3: Direct output in a list
                elif isinstance(response_json.get("outputs"), list) and len(response_json["outputs"]) > 0:
                    return response_json["outputs"][0]
                # Format 4: Single output field
                elif "output" in response_json:
                    return response_json["output"]
            # Format 5: Plain string
            elif isinstance(response_json, str):
                return response_json

            # Fallback: convert to string
            return str(response_json)

    # Prepare model_kwargs with parameters that should be passed to the endpoint
    endpoint_kwargs = {
        "temperature": llm_config.temperature,
        "top_p": llm_config.top_p,
        "max_new_tokens": llm_config.max_tokens,
    }

    # Add any additional model_kwargs from config
    if llm_config.model_kwargs:
        endpoint_kwargs.update(llm_config.model_kwargs)

    # Build the SagemakerEndpoint client
    config_dict = llm_config.model_dump(
        exclude={"type", "context_size", "thinking", "api_type", "temperature", "top_p", "max_tokens", "model_kwargs", "content_handler", "model"},
        by_alias=True,
        exclude_none=True,
    )

    base_llm = SagemakerEndpoint(
        **config_dict,
        content_handler=DefaultContentHandler(),
        model_kwargs=endpoint_kwargs,
    )

    # Wrap the LLM to make it compatible with chat-based agents
    # We need to convert it to a ChatModel-like interface
    class SageMakerChatWrapper(BaseChatModel):
        """Wrapper to make SageMaker LLM work like a ChatModel."""

        llm: BaseLLM = None  # Declare as a Pydantic field

        class Config:
            arbitrary_types_allowed = True

        @property
        def _llm_type(self) -> str:
            return "sagemaker_chat"

        def _generate(self, messages, stop=None, run_manager=None, **kwargs):
            # Convert messages to string prompt
            prompt = self._messages_to_prompt(messages)
            # Call the underlying LLM without triggering callbacks
            # Use _call instead of invoke to avoid nested callback issues
            try:
                text_response = self.llm._call(prompt, stop=stop, **kwargs)
            except AttributeError:
                # Fallback to invoke if _call not available
                text_response = self.llm.invoke(prompt, stop=stop, config={"callbacks": []}, **kwargs)
            # Convert to ChatGeneration
            message = AIMessage(content=text_response)
            generation = ChatGeneration(message=message)
            return ChatResult(generations=[generation])

        async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
            # Convert messages to string prompt
            prompt = self._messages_to_prompt(messages)
            # Call the underlying LLM without triggering callbacks
            # Use _acall instead of ainvoke to avoid nested callback issues
            try:
                text_response = await self.llm._acall(prompt, stop=stop, **kwargs)
            except AttributeError:
                # Fallback to ainvoke if _acall not available
                text_response = await self.llm.ainvoke(prompt, stop=stop, config={"callbacks": []}, **kwargs)
            # Convert to ChatGeneration
            message = AIMessage(content=text_response)
            generation = ChatGeneration(message=message)
            return ChatResult(generations=[generation])

        def _messages_to_prompt(self, messages) -> str:
            """Convert a list of messages to a string prompt using Mistral format."""
            prompt_parts = []

            # Mistral specific formatting
            for message in messages:
                if isinstance(message, SystemMessage):
                    # Mistral models handle system prompts at the beginning
                    prompt_parts.append(f"[INST] {message.content} [/INST]")
                elif isinstance(message, HumanMessage):
                    prompt_parts.append(f"[INST] {message.content} [/INST]")
                elif isinstance(message, AIMessage):
                    # AI responses don't need special formatting
                    prompt_parts.append(message.content)
                elif isinstance(message, BaseMessage):
                    prompt_parts.append(f"[INST] {message.content} [/INST]")
                else:
                    prompt_parts.append(str(message))

            # Join with newlines and ensure proper formatting
            return "\n".join(prompt_parts)

    client = SageMakerChatWrapper(llm=base_llm)

    yield _patch_llm_based_on_config(client, llm_config)


@register_llm_client(config_type=AzureOpenAIModelConfig, wrapper_type=LLMFrameworkEnum.LANGCHAIN)
async def azure_openai_langchain(llm_config: AzureOpenAIModelConfig, _builder: Builder):

    from langchain_openai import AzureChatOpenAI

    validate_no_responses_api(llm_config, LLMFrameworkEnum.LANGCHAIN)

    client = AzureChatOpenAI(
        **llm_config.model_dump(exclude={"type", "thinking", "api_type"}, by_alias=True, exclude_none=True))

    yield _patch_llm_based_on_config(client, llm_config)


@register_llm_client(config_type=NIMModelConfig, wrapper_type=LLMFrameworkEnum.LANGCHAIN)
async def nim_langchain(llm_config: NIMModelConfig, _builder: Builder):

    from langchain_nvidia_ai_endpoints import ChatNVIDIA

    validate_no_responses_api(llm_config, LLMFrameworkEnum.LANGCHAIN)

    # prefer max_completion_tokens over max_tokens
    client = ChatNVIDIA(
        **llm_config.model_dump(exclude={"type", "max_tokens", "thinking", "api_type"},
                                by_alias=True,
                                exclude_none=True),
        max_completion_tokens=llm_config.max_tokens,
    )

    yield _patch_llm_based_on_config(client, llm_config)


@register_llm_client(config_type=OpenAIModelConfig, wrapper_type=LLMFrameworkEnum.LANGCHAIN)
async def openai_langchain(llm_config: OpenAIModelConfig, _builder: Builder):

    from langchain_openai import ChatOpenAI

    if llm_config.api_type == APITypeEnum.RESPONSES:
        client = ChatOpenAI(stream_usage=True,
                            use_responses_api=True,
                            use_previous_response_id=True,
                            **llm_config.model_dump(
                                exclude={"type", "thinking", "api_type"},
                                by_alias=True,
                                exclude_none=True,
                            ))
    else:
        # If stream_usage is specified, it will override the default value of True.
        client = ChatOpenAI(stream_usage=True,
                            **llm_config.model_dump(
                                exclude={"type", "thinking", "api_type"},
                                by_alias=True,
                                exclude_none=True,
                            ))

    yield _patch_llm_based_on_config(client, llm_config)


@register_llm_client(config_type=LiteLlmModelConfig, wrapper_type=LLMFrameworkEnum.LANGCHAIN)
async def litellm_langchain(llm_config: LiteLlmModelConfig, _builder: Builder):

    from langchain_litellm import ChatLiteLLM

    validate_no_responses_api(llm_config, LLMFrameworkEnum.LANGCHAIN)

    client = ChatLiteLLM(
        **llm_config.model_dump(exclude={"type", "thinking", "api_type"}, by_alias=True, exclude_none=True))

    yield _patch_llm_based_on_config(client, llm_config)
