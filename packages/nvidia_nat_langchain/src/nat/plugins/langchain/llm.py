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

    # Create a default content handler for NIM-based endpoints
    class NIMContentHandler(LLMContentHandler):
        content_type = "application/json"
        accepts = "application/json"

        def transform_input(self, prompt: str, model_kwargs: dict) -> bytes:
            import json
            # Format for NVIDIA NIM models on SageMaker
            # NIM models expect OpenAI-compatible message format

            # Parse the prompt string back into messages
            # The wrapper formats messages as "System: ...\nUser: ...\n"
            messages = []

            # Split by the role prefixes
            lines = prompt.split('\n')
            current_role = None
            current_content = []

            for line in lines:
                if line.startswith('System: '):
                    if current_role and current_content:
                        messages.append({
                            "role": current_role,
                            "content": '\n'.join(current_content).strip()
                        })
                    current_role = "system"
                    current_content = [line[8:]]  # Remove "System: " prefix
                elif line.startswith('User: '):
                    if current_role and current_content:
                        messages.append({
                            "role": current_role,
                            "content": '\n'.join(current_content).strip()
                        })
                    current_role = "user"
                    current_content = [line[6:]]  # Remove "User: " prefix
                elif line.startswith('Assistant: '):
                    if current_role and current_content:
                        messages.append({
                            "role": current_role,
                            "content": '\n'.join(current_content).strip()
                        })
                    current_role = "assistant"
                    current_content = [line[11:]]  # Remove "Assistant: " prefix
                elif current_role:
                    current_content.append(line)

            # Add the last message
            if current_role and current_content:
                messages.append({
                    "role": current_role,
                    "content": '\n'.join(current_content).strip()
                })

            # If no messages were parsed (shouldn't happen), create a simple user message
            if not messages:
                messages = [{"role": "user", "content": prompt}]

            input_dict = {
                "messages": messages,
                "max_tokens": model_kwargs.get("max_new_tokens", model_kwargs.get("max_tokens", 1024)),
                "temperature": model_kwargs.get("temperature", 0.0),
                "top_p": model_kwargs.get("top_p", 1.0),
                "stream": False,  # Explicitly disable streaming
            }

            # Model parameter is REQUIRED for NIM endpoints
            # If not provided in model_kwargs, we need to raise a clear error
            if "model" in model_kwargs:
                input_dict["model"] = model_kwargs["model"]
            else:
                raise ValueError(
                    "The 'model' parameter is required for NIM-based SageMaker endpoints. "
                    "Please add it to your config under model_kwargs. Example:\n"
                    "  model_kwargs:\n"
                    "    model: \"your-model-name-here\"\n"
                    "Check your SageMaker endpoint documentation or NIM deployment to find the correct model name."
                )

            payload = json.dumps(input_dict).encode('utf-8')
            # logger.info(f"Sending request to SageMaker NIM endpoint: {json.dumps(input_dict, indent=2)}")
            return payload

        def transform_output(self, output: bytes) -> str:
            import json
            # Handle StreamingBody from boto3
            if hasattr(output, 'read'):
                output_bytes = output.read()
            else:
                output_bytes = output

            response_json = json.loads(output_bytes.decode("utf-8"))

            # Log the response for debugging
            #logger.info(f"Received response from SageMaker NIM endpoint: {json.dumps(response_json, indent=2)}")

            # Handle NIM/OpenAI-compatible response format
            if isinstance(response_json, dict):
                # Check for error response
                if "error" in response_json:
                    logger.error(f"Error response from endpoint: {json.dumps(response_json, indent=2)}")
                    error_msg = response_json.get("error", {})
                    if isinstance(error_msg, dict):
                        raise ValueError(f"Endpoint error: {error_msg.get('message', str(error_msg))}")
                    else:
                        raise ValueError(f"Endpoint error: {error_msg}")

                # Format 1: OpenAI-style choices array (NIM format)
                if "choices" in response_json and len(response_json["choices"]) > 0:
                    choice = response_json["choices"][0]
                    if "message" in choice:
                        return choice["message"].get("content", "")
                    elif "text" in choice:
                        return choice["text"]
                # Format 2: Direct content field
                elif "content" in response_json:
                    return response_json["content"]
                # Format 3: Generated text field (fallback)
                elif "generated_text" in response_json:
                    return response_json["generated_text"]
            # Format 4: Plain string
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
        content_handler=NIMContentHandler(),
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
            except Exception as e:
                logger.error(f"Error calling SageMaker endpoint: {e}")
                # Try to extract more details from the error
                if hasattr(e, 'response'):
                    logger.error(f"Error response: {e.response}")
                raise
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
            except Exception as e:
                logger.error(f"Error calling SageMaker endpoint: {e}")
                # Try to extract more details from the error
                if hasattr(e, 'response'):
                    logger.error(f"Error response: {e.response}")
                raise
            # Convert to ChatGeneration
            message = AIMessage(content=text_response)
            generation = ChatGeneration(message=message)
            return ChatResult(generations=[generation])

        def _messages_to_prompt(self, messages) -> str:
            """Convert a list of messages to a string prompt.

            For NIM models, we create a conversational format without special tags.
            The actual message formatting will be handled by the content handler.
            """
            prompt_parts = []

            for message in messages:
                if isinstance(message, SystemMessage):
                    # Include system message as context
                    prompt_parts.append(f"System: {message.content}")
                elif isinstance(message, HumanMessage):
                    prompt_parts.append(f"User: {message.content}")
                elif isinstance(message, AIMessage):
                    prompt_parts.append(f"Assistant: {message.content}")
                elif isinstance(message, BaseMessage):
                    prompt_parts.append(f"User: {message.content}")
                else:
                    prompt_parts.append(str(message))

            # Join with newlines
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
