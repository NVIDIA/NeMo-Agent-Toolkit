# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Runtime LLM clients for raw OpenAI SDK usage.

This module wires the OpenAI SDK into the NVIDIA NeMo Agent Toolkit runtime
so workflows can call models directly while still emitting intermediate step
events and honoring toolkit configuration defaults.
"""

import logging
import os
from collections.abc import AsyncGenerator
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING
from typing import Any
from typing import TypeAlias
from typing import TypeVar
from typing import cast
from uuid import uuid4

from openai import AsyncOpenAI
from openai import AsyncStream
from openai.types.chat import ChatCompletion
from openai.types.chat import ChatCompletionChunk
from openai.types.chat import ChatCompletionMessageParam
from openai.types.chat import CompletionCreateParams
from openai.types.responses import Response
from openai.types.responses import ResponseCreateParams
from openai.types.responses import ResponseInputParam
from openai.types.responses import ResponseStreamEvent
from pydantic import BaseModel

from nat.builder.builder import Builder
from nat.builder.context import Context
from nat.builder.framework_enum import LLMFrameworkEnum
from nat.cli.register_workflow import register_llm_client
from nat.data_models.common import get_secret_value
from nat.data_models.intermediate_step import IntermediateStepPayload
from nat.data_models.intermediate_step import IntermediateStepType
from nat.data_models.intermediate_step import StreamEventData
from nat.data_models.retry_mixin import RetryMixin
from nat.llm.dynamo_llm import DynamoModelConfig
from nat.llm.dynamo_llm import create_httpx_client_with_dynamo_hooks
from nat.llm.litellm_llm import LiteLlmModelConfig
from nat.llm.nim_llm import NIMModelConfig
from nat.llm.openai_llm import OpenAIModelConfig
from nat.llm.utils.hooks import create_metadata_injection_client
from nat.profiler.prediction_trie import load_prediction_trie
from nat.profiler.prediction_trie.trie_lookup import PredictionTrieLookup
from nat.utils.exception_handlers.automatic_retries import patch_with_retry

if TYPE_CHECKING:
    import httpx

logger = logging.getLogger(__name__)

ClientT = TypeVar("ClientT")
ChatMessages: TypeAlias = list[ChatCompletionMessageParam]
LLMInput: TypeAlias = ChatMessages | ResponseInputParam
StreamedChunk: TypeAlias = ChatCompletionChunk | ResponseStreamEvent
StreamPayload: TypeAlias = StreamedChunk | str
ChatCompletionStream: TypeAlias = AsyncGenerator[ChatCompletionChunk, None]
ResponseStream: TypeAlias = AsyncGenerator[ResponseStreamEvent, None]
ChatCompletionOutput: TypeAlias = ChatCompletion | ChatCompletionStream
ResponseOutput: TypeAlias = Response | ResponseStream
LLMOutputPayload: TypeAlias = ChatCompletion | Response | str | list[ChatCompletionChunk] | list[ResponseStreamEvent]


def _apply_retry(client: ClientT, llm_config: BaseModel) -> ClientT:
    """Wrap a runtime client with retry handling when configured.

    Args:
        client: Runtime client instance to wrap.
        llm_config: Configuration object that may include retry settings.

    Returns:
        The original client when retries are disabled, otherwise a retry-enabled
        wrapper.
    """
    if isinstance(llm_config, RetryMixin):
        return patch_with_retry(
            client,
            retries=llm_config.num_retries,
            retry_codes=llm_config.retry_on_status_codes,
            retry_on_messages=llm_config.retry_on_errors,
        )
    return client


def _build_default_params(llm_config: BaseModel, extra_exclude: set[str] | None = None) -> dict[str, object]:
    """Build OpenAI SDK parameters from a model config.

    Args:
        llm_config: Configuration object for the model.
        extra_exclude: Optional set of config field names to omit.

    Returns:
        Dictionary of parameters ready for OpenAI SDK calls, filtered to exclude
        non-API fields and unset values.
    """
    exclude = {"type", "api_type", "thinking", "api_key", "base_url", "model_name", "max_retries"}
    if extra_exclude:
        exclude |= extra_exclude
    params = llm_config.model_dump(
        exclude=exclude,
        by_alias=True,
        exclude_none=True,
        exclude_unset=True,
    )
    params.pop("stream_usage", None)
    return params


def _ensure_stream_usage(params: dict[str, object], stream: bool) -> None:
    """Ensure usage metadata is requested for streaming responses.

    Args:
        params: Parameter dictionary that will be sent to the OpenAI SDK.
        stream: Whether streaming is enabled for the call.
    """
    if not stream:
        return
    stream_options_value = params.get("stream_options")
    stream_options = dict(stream_options_value) if isinstance(stream_options_value, Mapping) else {}
    stream_options["include_usage"] = True
    params["stream_options"] = stream_options


def _extract_text_from_chunk(chunk: StreamedChunk | dict[str, object]) -> str | None:
    """Extract text content from an OpenAI SDK streaming chunk.

    Args:
        chunk: Streaming chunk object or dictionary from the OpenAI SDK.

    Returns:
        The extracted text when available, otherwise ``None``.
    """
    if isinstance(chunk, dict):
        choices = chunk.get("choices")
        if isinstance(choices, list) and choices:
            choice = choices[0]
            delta = choice.get("delta") if isinstance(choice, dict) else None
            content = delta.get("content") if isinstance(delta, dict) else None
            if isinstance(content, str):
                return content
        output_text = chunk.get("output_text")
        if isinstance(output_text, str):
            return output_text
        return None

    choices = getattr(chunk, "choices", None)
    if choices:
        delta = getattr(choices[0], "delta", None)
        content = getattr(delta, "content", None) if delta is not None else None
        if isinstance(content, str):
            return content

    output_text = getattr(chunk, "output_text", None)
    if isinstance(output_text, str):
        return output_text

    return None


class RuntimeOpenAIClient:
    """Wrapper around the OpenAI SDK client with toolkit defaults applied.

    The wrapper applies model defaults, emits intermediate step events, and
    normalizes streaming usage metadata for NVIDIA NeMo Agent Toolkit workflows.
    """

    def __init__(self, client: AsyncOpenAI, *, model_name: str, default_params: dict[str, object]) -> None:
        """Initialize the runtime client wrapper.

        Args:
            client: Underlying OpenAI SDK client.
            model_name: Default model identifier to use when unspecified.
            default_params: Default request parameters applied to calls.
        """
        self._client = client
        self._model_name = model_name
        self._default_params = dict(default_params)

    @property
    def client(self) -> AsyncOpenAI:
        """Return the underlying OpenAI SDK client."""
        return self._client

    def _build_chat_params(self, **kwargs: object) -> CompletionCreateParams:
        """Merge default parameters with chat completion overrides."""
        params = {**self._default_params, **kwargs}
        params.setdefault("model", self._model_name)
        stream = bool(params.get("stream"))
        _ensure_stream_usage(params, stream)
        return cast(CompletionCreateParams, params)

    def _build_response_params(self, **kwargs: object) -> ResponseCreateParams:
        """Merge default parameters with response overrides."""
        params = {**self._default_params, **kwargs}
        params.setdefault("model", self._model_name)
        stream = bool(params.get("stream"))
        _ensure_stream_usage(params, stream)
        return cast(ResponseCreateParams, params)

    def _emit_llm_step(self,
                       *,
                       event_type: IntermediateStepType,
                       step_id: str,
                       input_data: LLMInput,
                       output_data: LLMOutputPayload | None) -> None:
        """Emit an LLM lifecycle step to the intermediate step manager.

        Args:
            event_type: Step event type to emit.
            step_id: Unique identifier for this LLM invocation.
            input_data: Input data sent to the LLM call.
            output_data: Optional response payload.
        """
        payload = IntermediateStepPayload(
            event_type=event_type,
            framework=LLMFrameworkEnum.RUNTIME,
            name=self._model_name,
            UUID=step_id,
            data=StreamEventData(input=input_data, output=output_data),
        )
        Context.get().intermediate_step_manager.push_intermediate_step(payload)

    def _emit_llm_chunk(self, *, step_id: str, input_data: LLMInput, chunk: StreamPayload) -> None:
        """Emit a streaming token/chunk step.

        Args:
            step_id: Unique identifier for this LLM invocation.
            input_data: Input data sent to the LLM call.
            chunk: Streaming chunk payload.
        """
        payload = IntermediateStepPayload(
            event_type=IntermediateStepType.LLM_NEW_TOKEN,
            framework=LLMFrameworkEnum.RUNTIME,
            name=self._model_name,
            UUID=step_id,
            data=StreamEventData(input=input_data, chunk=chunk),
        )
        Context.get().intermediate_step_manager.push_intermediate_step(payload)

    async def chat_completions(self, messages: ChatMessages, **kwargs: object) -> ChatCompletionOutput:
        """Create a chat completion using the OpenAI SDK.

        This method emits start/end intermediate steps and, when streaming,
        emits token events for each chunk.

        Args:
            messages: Chat messages in OpenAI SDK format.
            kwargs: Additional OpenAI SDK parameters.

        Returns:
            OpenAI SDK response object, or an async generator when streaming.
        """
        params = self._build_chat_params(**kwargs)
        step_id = str(uuid4())
        self._emit_llm_step(event_type=IntermediateStepType.LLM_START,
                            step_id=step_id,
                            input_data=messages,
                            output_data=None)
        if params.get("stream"):
            stream = cast(AsyncStream[ChatCompletionChunk],
                          await self._client.chat.completions.create(messages=messages, **cast(dict[str, Any], params)))

            async def _stream() -> ChatCompletionStream:
                chunks: list[ChatCompletionChunk] = []
                text_parts: list[str] = []
                try:
                    async for chunk in stream:
                        chunks.append(chunk)
                        text = _extract_text_from_chunk(chunk)
                        if text:
                            text_parts.append(text)
                        chunk_payload = text if text is not None else chunk
                        self._emit_llm_chunk(step_id=step_id, input_data=messages, chunk=chunk_payload)
                        yield chunk
                finally:
                    output_data: LLMOutputPayload = "".join(text_parts) if text_parts else chunks
                    self._emit_llm_step(event_type=IntermediateStepType.LLM_END,
                                        step_id=step_id,
                                        input_data=messages,
                                        output_data=output_data)

            return _stream()

        response = cast(ChatCompletion,
                        await self._client.chat.completions.create(messages=messages, **cast(dict[str, Any], params)))
        self._emit_llm_step(event_type=IntermediateStepType.LLM_END,
                            step_id=step_id,
                            input_data=messages,
                            output_data=response)
        return response

    async def responses(self, input_data: ResponseInputParam, **kwargs: object) -> ResponseOutput:
        """Create a response using the OpenAI SDK.

        This method emits start/end intermediate steps and, when streaming,
        emits token events for each chunk.

        Args:
            input_data: Input payload for the Responses API.
            **kwargs: Additional OpenAI SDK parameters.

        Returns:
            OpenAI SDK response object, or an async generator when streaming.
        """
        params = self._build_response_params(**kwargs)
        step_id = str(uuid4())
        self._emit_llm_step(event_type=IntermediateStepType.LLM_START,
                            step_id=step_id,
                            input_data=input_data,
                            output_data=None)
        if params.get("stream"):
            stream = cast(AsyncStream[ResponseStreamEvent],
                          await self._client.responses.create(input=input_data, **cast(dict[str, Any], params)))

            async def _stream() -> ResponseStream:
                chunks: list[ResponseStreamEvent] = []
                text_parts: list[str] = []
                try:
                    async for chunk in stream:
                        chunks.append(chunk)
                        text = _extract_text_from_chunk(chunk)
                        if text:
                            text_parts.append(text)
                        chunk_payload = text if text is not None else chunk
                        self._emit_llm_chunk(step_id=step_id, input_data=input_data, chunk=chunk_payload)
                        yield chunk
                finally:
                    output_data: LLMOutputPayload = "".join(text_parts) if text_parts else chunks
                    self._emit_llm_step(event_type=IntermediateStepType.LLM_END,
                                        step_id=step_id,
                                        input_data=input_data,
                                        output_data=output_data)

            return _stream()

        response = cast(Response, await self._client.responses.create(input=input_data, **cast(dict[str, Any], params)))
        self._emit_llm_step(event_type=IntermediateStepType.LLM_END,
                            step_id=step_id,
                            input_data=input_data,
                            output_data=response)
        return response


def _build_openai_client(*, api_key: str | None, base_url: str | None,
                         http_async_client: httpx.AsyncClient) -> AsyncOpenAI:
    """Construct an OpenAI SDK client with the provided HTTP transport.

    Args:
        api_key: API key for the target endpoint.
        base_url: Optional base URL override for OpenAI-compatible endpoints.
        http_async_client: Async HTTP client used by the SDK.

    Returns:
        Configured `AsyncOpenAI` client instance.
    """
    return AsyncOpenAI(
        api_key=api_key,
        base_url=base_url,
        http_client=http_async_client,
    )


@register_llm_client(config_type=OpenAIModelConfig, wrapper_type=LLMFrameworkEnum.RUNTIME)
async def openai_runtime(llm_config: OpenAIModelConfig, _builder: Builder) -> AsyncGenerator[RuntimeOpenAIClient, None]:
    """Build a runtime OpenAI SDK client for OpenAI-hosted models.

    Args:
        llm_config: OpenAI model configuration.
        _builder: Builder instance (unused).

    Yields:
        Configured runtime client for OpenAI-hosted models.
    """
    api_key = get_secret_value(llm_config.api_key) or os.getenv("OPENAI_API_KEY")
    async with create_metadata_injection_client() as http_async_client:
        oai_client = _build_openai_client(
            api_key=api_key,
            base_url=llm_config.base_url,
            http_async_client=http_async_client,
        )
        runtime_client = RuntimeOpenAIClient(
            oai_client,
            model_name=llm_config.model_name,
            default_params=_build_default_params(llm_config),
        )
        yield _apply_retry(runtime_client, llm_config)


@register_llm_client(config_type=NIMModelConfig, wrapper_type=LLMFrameworkEnum.RUNTIME)
async def nim_runtime(llm_config: NIMModelConfig, _builder: Builder) -> AsyncGenerator[RuntimeOpenAIClient, None]:
    """Build a runtime OpenAI SDK client for NVIDIA NIM endpoints.

    Args:
        llm_config: NIM model configuration.
        _builder: Builder instance (unused).

    Yields:
        Configured runtime client for NVIDIA NIM endpoints.
    """
    base_url = llm_config.base_url or "https://integrate.api.nvidia.com/v1"
    api_key = get_secret_value(llm_config.api_key) or os.getenv("NVIDIA_API_KEY")
    if llm_config.base_url and llm_config.base_url.strip() and api_key is None:
        api_key = "dummy-api-key"

    async with create_metadata_injection_client() as http_async_client:
        oai_client = _build_openai_client(
            api_key=api_key,
            base_url=base_url,
            http_async_client=http_async_client,
        )
        runtime_client = RuntimeOpenAIClient(
            oai_client,
            model_name=llm_config.model_name,
            default_params=_build_default_params(llm_config),
        )
        yield _apply_retry(runtime_client, llm_config)


@register_llm_client(config_type=DynamoModelConfig, wrapper_type=LLMFrameworkEnum.RUNTIME)
async def dynamo_runtime(llm_config: DynamoModelConfig, _builder: Builder) -> AsyncGenerator[RuntimeOpenAIClient, None]:
    """Build a runtime OpenAI SDK client for Dynamo endpoints.

    Args:
        llm_config: Dynamo model configuration.
        _builder: Builder instance (unused).

    Yields:
        Configured runtime client for Dynamo endpoints.
    """
    api_key = get_secret_value(llm_config.api_key) or os.getenv("OPENAI_API_KEY")
    prediction_lookup: PredictionTrieLookup | None = None
    if llm_config.prediction_trie_path:
        try:
            trie_path = Path(llm_config.prediction_trie_path)
            trie = load_prediction_trie(trie_path)
            prediction_lookup = PredictionTrieLookup(trie)
        except FileNotFoundError:
            logger.warning("Prediction trie file not found: %s", llm_config.prediction_trie_path)
        except Exception as exc:
            logger.warning("Failed to load prediction trie: %s", exc)

    http_async_client = create_httpx_client_with_dynamo_hooks(
        prefix_template=llm_config.prefix_template,
        total_requests=llm_config.prefix_total_requests,
        osl=llm_config.prefix_osl,
        iat=llm_config.prefix_iat,
        timeout=llm_config.request_timeout,
        prediction_lookup=prediction_lookup,
    )
    async with http_async_client:
        oai_client = _build_openai_client(
            api_key=api_key,
            base_url=llm_config.base_url,
            http_async_client=http_async_client,
        )
        runtime_client = RuntimeOpenAIClient(
            oai_client,
            model_name=llm_config.model_name,
            default_params=_build_default_params(
                llm_config,
                extra_exclude=set(DynamoModelConfig.get_dynamo_field_names()),
            ),
        )
        yield _apply_retry(runtime_client, llm_config)


@register_llm_client(config_type=LiteLlmModelConfig, wrapper_type=LLMFrameworkEnum.RUNTIME)
async def litellm_runtime(llm_config: LiteLlmModelConfig,
                          _builder: Builder) -> AsyncGenerator[RuntimeOpenAIClient, None]:
    """Build a runtime OpenAI SDK client for LiteLLM endpoints.

    Args:
        llm_config: LiteLLM model configuration.
        _builder: Builder instance (unused).

    Yields:
        Configured runtime client for LiteLLM endpoints.
    """
    api_key = get_secret_value(llm_config.api_key) or os.getenv("OPENAI_API_KEY")
    async with create_metadata_injection_client() as http_async_client:
        oai_client = _build_openai_client(
            api_key=api_key,
            base_url=llm_config.base_url,
            http_async_client=http_async_client,
        )
        runtime_client = RuntimeOpenAIClient(
            oai_client,
            model_name=llm_config.model_name,
            default_params=_build_default_params(llm_config),
        )
        yield _apply_retry(runtime_client, llm_config)
