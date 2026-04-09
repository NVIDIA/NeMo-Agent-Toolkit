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
"""Profiler callback handler for AG2.

Patches AG2's OpenAIWrapper to capture LLM call timings and
token usage for the profiling pipeline.

Supported Methods
-----------------
- ``OpenAIWrapper.create``: LLM completions (sync — AG2's only LLM call path)
"""

import logging
import threading
import time
from collections.abc import Callable
from typing import Any

from nat.builder.context import Context
from nat.builder.framework_enum import LLMFrameworkEnum
from nat.data_models.intermediate_step import IntermediateStepPayload
from nat.data_models.intermediate_step import IntermediateStepType
from nat.data_models.intermediate_step import StreamEventData
from nat.data_models.intermediate_step import UsageInfo
from nat.data_models.profiler_callback import BaseProfilerCallback
from nat.data_models.token_usage import TokenUsageBaseModel

logger = logging.getLogger(__name__)


class AG2ProfilerHandler(BaseProfilerCallback):
    """Instruments AG2 agents for profiling.

    Monkey-patches AG2's ``OpenAIWrapper.create`` to collect telemetry
    data including token usage, inputs, outputs, and timing information.

    Example:
        >>> handler = AG2ProfilerHandler()
        >>> handler.instrument()
        >>> # ... run AG2 workflow ...
        >>> handler.uninstrument()
    """

    def __init__(self) -> None:
        """Initialize the AG2ProfilerHandler."""
        super().__init__()
        self._lock = threading.Lock()
        self.last_call_ts = time.time()
        self.step_manager = Context.get().intermediate_step_manager
        self._original_create: Callable[..., Any] | None = None
        self._instrumented = False

    def instrument(self) -> None:
        """Monkey-patch AG2's LLM calling to emit profiling events.

        Patches ``OpenAIWrapper.create`` if available.
        Does nothing if already instrumented or if imports fail.
        """
        if self._instrumented:
            logger.debug("AG2ProfilerHandler already instrumented; skipping.")
            return

        try:
            from autogen.oai.client import OpenAIWrapper
        except ImportError:
            logger.debug("autogen.oai.client not available; skipping AG2 instrumentation")
            return

        self._original_create = OpenAIWrapper.create
        OpenAIWrapper.create = self._create_llm_wrapper(self._original_create)
        logger.debug("Patched OpenAIWrapper.create")

        self._instrumented = True
        logger.debug("AG2ProfilerHandler instrumentation applied successfully.")

    def uninstrument(self) -> None:
        """Restore original AG2 methods.

        Should be called to clean up monkey patches, especially in test environments.
        """
        try:
            if self._original_create is not None:
                from autogen.oai.client import OpenAIWrapper

                OpenAIWrapper.create = self._original_create
                self._original_create = None
                logger.debug("Restored OpenAIWrapper.create")

            self._instrumented = False
            logger.debug("AG2ProfilerHandler uninstrumented successfully.")
        except Exception:
            logger.exception("Failed to uninstrument AG2ProfilerHandler")

    # Keep legacy aliases for backwards compatibility
    patch = instrument
    unpatch = uninstrument

    def _extract_model_name(self, kwargs: dict[str, Any]) -> str:
        """Extract model name from call kwargs.

        Args:
            kwargs: The keyword arguments passed to OpenAIWrapper.create

        Returns:
            str: Model name or empty string if not found
        """
        return str(kwargs.get("model", ""))

    def _extract_input_text(self, kwargs: dict[str, Any]) -> str:
        """Extract sanitized input from call kwargs.

        Args:
            kwargs: The keyword arguments passed to OpenAIWrapper.create

        Returns:
            str: String representation of sanitized input
        """
        messages = kwargs.get("messages")
        try:
            message_count = len(messages)
        except TypeError:
            message_count = 0
        return str({
            "model": kwargs.get("model", ""),
            "message_count": message_count,
        })

    def _create_llm_wrapper(self, original_func: Callable[..., Any]) -> Callable[..., Any]:
        """Create wrapper for LLM calls.

        AG2's ``OpenAIWrapper.create`` is synchronous, so the wrapper is also
        synchronous.

        Args:
            original_func: Original create method to wrap

        Returns:
            Callable: Wrapped function with profiling
        """
        handler = self

        def wrapped_llm_call(wrapper_self: Any, *args: Any, **kwargs: Any) -> Any:
            now = time.time()
            with handler._lock:
                seconds_between_calls = int(now - handler.last_call_ts)

            model_name = handler._extract_model_name(kwargs)
            model_input = handler._extract_input_text(kwargs)

            # Push LLM_START event
            start_payload = IntermediateStepPayload(
                event_type=IntermediateStepType.LLM_START,
                framework=LLMFrameworkEnum.AG2,
                name=model_name,
                data=StreamEventData(input=model_input),
                usage_info=UsageInfo(
                    token_usage=TokenUsageBaseModel(),
                    num_llm_calls=1,
                    seconds_between_calls=seconds_between_calls,
                ),
            )
            start_uuid = start_payload.UUID
            handler.step_manager.push_intermediate_step(start_payload)

            try:
                result = original_func(wrapper_self, *args, **kwargs)
            except Exception as e:
                logger.error("Error during LLM call: %s", e)
                err_time = time.time()
                handler.step_manager.push_intermediate_step(
                    IntermediateStepPayload(
                        event_type=IntermediateStepType.LLM_END,
                        span_event_timestamp=err_time,
                        framework=LLMFrameworkEnum.AG2,
                        name=model_name,
                        data=StreamEventData(
                            input=model_input,
                            output=f"error: {type(e).__name__}",
                        ),
                        usage_info=UsageInfo(token_usage=TokenUsageBaseModel(), ),
                        UUID=start_uuid,
                    ))
                with handler._lock:
                    handler.last_call_ts = err_time
                raise

            # Push LLM_END event
            end_time = time.time()
            handler.step_manager.push_intermediate_step(
                IntermediateStepPayload(
                    event_type=IntermediateStepType.LLM_END,
                    span_event_timestamp=end_time,
                    framework=LLMFrameworkEnum.AG2,
                    name=model_name,
                    data=StreamEventData(
                        input=model_input,
                        output="completed",
                    ),
                    usage_info=UsageInfo(
                        token_usage=TokenUsageBaseModel(),
                        num_llm_calls=1,
                    ),
                    UUID=start_uuid,
                ))
            with handler._lock:
                handler.last_call_ts = end_time

            return result

        return wrapped_llm_call
