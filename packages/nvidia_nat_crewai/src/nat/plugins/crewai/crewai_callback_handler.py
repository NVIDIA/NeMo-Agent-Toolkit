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

import copy
import logging
import threading
import time
from collections.abc import Callable
from typing import Any

import litellm
from crewai.tools import tool_usage

from nat.builder.context import Context
from nat.builder.framework_enum import LLMFrameworkEnum
from nat.data_models.intermediate_step import IntermediateStepPayload
from nat.data_models.intermediate_step import IntermediateStepType
from nat.data_models.intermediate_step import StreamEventData
from nat.data_models.intermediate_step import TraceMetadata
from nat.data_models.intermediate_step import UsageInfo
from nat.data_models.profiler_callback import BaseProfilerCallback
from nat.data_models.token_usage import TokenUsageBaseModel

logger = logging.getLogger(__name__)


def _record_end_safely(push_end: Callable[[], None]) -> None:
    """Record the end event of a failed call without letting a recording error replace the call's own error.

    Args:
        push_end: Pushes the end event that closes the failed call's start event.
    """
    try:
        push_end()
    except Exception:
        logger.exception("Error recording the end of a failed call")


class _ProfiledStream:
    """
    Passes a streamed ``litellm.completion`` response through to the caller and collects its chunks.

    ``on_end`` is called once with the chunks received so far, when the stream is exhausted, raises or is closed.
    Other attributes are read from the wrapped stream.
    """

    def __init__(self, stream: Any, on_end: Callable[[list[Any]], None]):
        """Wrap ``stream`` and call ``on_end`` with its chunks once it ends."""
        self._stream = stream
        self._on_end = on_end
        self._chunks: list[Any] = []
        self._ended = False

    def __getattr__(self, name: str) -> Any:
        """Read any other attribute from the wrapped stream."""
        return getattr(self._stream, name)

    def __iter__(self) -> "_ProfiledStream":
        """Return this wrapper as the iterator."""
        return self

    def __next__(self) -> Any:
        """Return the next chunk of the wrapped stream, recording the end of the stream when it stops."""
        try:
            chunk = next(self._stream)
        except BaseException:
            # StopIteration when the stream is exhausted, or the error that ended it
            self._end()
            raise
        self._chunks.append(chunk)
        return chunk

    def close(self) -> None:
        """Close the wrapped stream if it has ``close``, then record the end of a stream the caller stopped reading."""
        try:
            close = getattr(self._stream, "close", None)
            if callable(close):
                close()
        finally:
            self._end()

    def _end(self) -> None:
        """Call ``on_end`` once with the chunks received so far."""
        if self._ended:
            return
        self._ended = True
        try:
            self._on_end(self._chunks)
        except Exception:
            logger.exception("Error recording the end of a streamed LLM call")


class CrewAIProfilerHandler(BaseProfilerCallback):
    """
    A callback manager/handler for CrewAI that intercepts calls to:
      - ToolUsage._use
      - LLM Calls

    to collect usage statistics (tokens, inputs, outputs, time intervals, etc.)
    and store them in NAT's usage_stats queue for subsequent analysis.
    """

    def __init__(self) -> None:
        super().__init__()
        self._lock = threading.Lock()
        self.last_call_ts = time.time()
        self.step_manager = Context.get().intermediate_step_manager

        # Original references to CrewAI methods (for uninstrumenting if needed)
        self._original_tool_use = None
        self._original_llm_call = None

    def instrument(self) -> None:
        """
        Monkey-patch the relevant CrewAI methods with usage-stat collection logic.
        Assumes the 'crewai' library is installed.
        """

        # Save the originals
        self._original_tool_use = getattr(tool_usage.ToolUsage, "_use", None)
        self._original_llm_call = getattr(litellm, "completion", None)

        # Patch if available
        if self._original_tool_use:
            tool_usage.ToolUsage._use = self._tool_use_monkey_patch()

        if self._original_llm_call:
            litellm.completion = self._llm_call_monkey_patch()

        logger.debug("CrewAIProfilerHandler instrumentation applied successfully.")

    def _tool_use_monkey_patch(self) -> Callable[..., Any]:
        """
        Returns a function that wraps calls to ToolUsage._use(...) with usage-logging.
        """
        original_func = self._original_tool_use

        def wrapped_tool_use(tool_usage_instance, *args, **kwargs) -> Any:
            """
            Replicates _tool_use_wrapper logic without wrapt: collects usage stats,
            calls the original, and captures output stats.
            """
            now = time.time()
            tool_name = ""

            try:
                tool_info = kwargs.get("tool", "")

                if tool_info:
                    tool_name = tool_info.name
            except Exception as e:
                logger.exception("Error getting tool name: %s", e)

            # Pre-call usage event
            stats = IntermediateStepPayload(event_type=IntermediateStepType.TOOL_START,
                                            framework=LLMFrameworkEnum.CREWAI,
                                            name=tool_name,
                                            data=StreamEventData(),
                                            metadata=TraceMetadata(tool_inputs={
                                                "args": args, "kwargs": dict(kwargs)
                                            }),
                                            usage_info=UsageInfo(token_usage=TokenUsageBaseModel()))

            self.step_manager.push_intermediate_step(stats)

            self.last_call_ts = now

            def push_tool_end(output: str = "") -> None:
                """Record the end event that closes this call's TOOL_START."""
                usage_stat = IntermediateStepPayload(
                    event_type=IntermediateStepType.TOOL_END,
                    UUID=stats.UUID,
                    span_event_timestamp=now,
                    framework=LLMFrameworkEnum.CREWAI,
                    name=tool_name,
                    data=StreamEventData(input={
                        "args": args, "kwargs": dict(kwargs)
                    }, output=output),
                    metadata=TraceMetadata(tool_outputs={"result": output}),
                    usage_info=UsageInfo(token_usage=TokenUsageBaseModel()),
                )

                self.step_manager.push_intermediate_step(usage_stat)

            try:
                # Call the original _use(...)
                result = original_func(tool_usage_instance, *args, **kwargs)
            except BaseException as e:
                # Close the tool step, then let CrewAI see the tool's error
                if isinstance(e, Exception):
                    logger.error("ToolUsage._use error: %s", e)
                _record_end_safely(push_tool_end)
                raise

            push_tool_end(str(result))

            return result

        return wrapped_tool_use

    def _llm_call_monkey_patch(self) -> Callable[..., Any]:
        """
        Returns a function that wraps calls to litellm.completion(...) with usage-logging.
        """
        original_func = self._original_llm_call

        def wrapped_llm_call(*args, **kwargs) -> Any:
            """
            Replicates _llm_call_wrapper logic without wrapt: collects usage stats,
            calls the original, and captures output stats.
            """

            now = time.time()
            seconds_between_calls = int(now - self.last_call_ts)
            model_name = kwargs.get('model', "")

            model_input = []
            try:
                for message in kwargs.get('messages', []):
                    content = message.get('content', "")
                    model_input.append("" if content is None else str(content))
            except Exception as e:
                logger.exception("Error getting model input: %s", e)

            model_input = "".join(model_input)

            # Record the start event
            input_stats = IntermediateStepPayload(
                event_type=IntermediateStepType.LLM_START,
                framework=LLMFrameworkEnum.CREWAI,
                name=model_name,
                data=StreamEventData(input=model_input),
                metadata=TraceMetadata(chat_inputs=copy.deepcopy(kwargs.get('messages', []))),
                usage_info=UsageInfo(token_usage=TokenUsageBaseModel(),
                                     num_llm_calls=1,
                                     seconds_between_calls=seconds_between_calls))

            self.step_manager.push_intermediate_step(input_stats)

            def push_llm_end(model_output: str = "",
                             chat_response: dict[str, Any] | None = None,
                             token_usage: TokenUsageBaseModel | None = None) -> None:
                """Record the end event that closes this call's LLM_START."""
                output_stats = IntermediateStepPayload(event_type=IntermediateStepType.LLM_END,
                                                       UUID=input_stats.UUID,
                                                       span_event_timestamp=now,
                                                       framework=LLMFrameworkEnum.CREWAI,
                                                       name=model_name,
                                                       data=StreamEventData(input=model_input, output=model_output),
                                                       metadata=TraceMetadata(chat_responses=chat_response),
                                                       usage_info=UsageInfo(
                                                           token_usage=token_usage or TokenUsageBaseModel(),
                                                           num_llm_calls=1,
                                                           seconds_between_calls=seconds_between_calls))

                self.step_manager.push_intermediate_step(output_stats)

            def push_response_end(output: Any) -> None:
                """Record the end event for a call from its response."""
                model_output = []
                try:
                    for choice in output.choices:
                        if hasattr(choice, 'message') and hasattr(choice.message, 'content'):
                            content = choice.message.content or ""
                        elif hasattr(choice, 'model_extra') and 'message' in choice.model_extra:
                            msg = choice.model_extra["message"]
                            content = msg.get('content', "") or ""
                        else:
                            content = ""
                        model_output.append(str(content))
                except Exception as e:
                    logger.exception("Error getting model output: %s", e)

                model_output = "".join(model_output)

                chat_response = None
                token_usage = TokenUsageBaseModel()
                try:
                    if output.choices:
                        chat_response = output.choices[0].model_dump()
                    model_extra = getattr(output, 'model_extra', None) or {}
                    if 'usage' in model_extra:
                        token_usage = TokenUsageBaseModel(**model_extra['usage'].model_dump())
                except Exception as e:
                    logger.exception("Error getting model usage: %s", e)

                # Record the end event
                push_llm_end(model_output, chat_response, token_usage)

            def push_stream_end(chunks: list[Any]) -> None:
                """Record the end event for a streamed call from the chunks it yielded."""
                if not chunks:
                    push_llm_end()
                    return
                try:
                    response = litellm.stream_chunk_builder(chunks, messages=kwargs.get('messages'))
                except Exception as e:
                    logger.exception("Error combining the streamed chunks: %s", e)
                    push_llm_end()
                    return
                push_response_end(response)

            try:
                # Call the original litellm.completion(...)
                output = original_func(*args, **kwargs)
            except BaseException:
                # Close the LLM step, then let CrewAI see the provider's error
                _record_end_safely(push_llm_end)
                raise

            if kwargs.get('stream'):
                # With stream=True LiteLLM returns a stream that CrewAI reads after this call returns. Record the end
                # event once the stream is exhausted, fails or is closed, with the output and usage combined from its
                # chunks.
                return _ProfiledStream(output, push_stream_end)

            push_response_end(output)

            # (Note: the original code did NOT update self.last_call_ts here)
            return output

        return wrapped_llm_call
