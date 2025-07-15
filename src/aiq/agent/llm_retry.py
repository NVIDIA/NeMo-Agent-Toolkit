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

"""Retry utilities for LLM calls with exponential backoff and jitter."""

import asyncio
import logging
import random
from collections.abc import Callable
from typing import Any

from langchain_core.messages.base import BaseMessage
from langchain_core.runnables import RunnableConfig

logger = logging.getLogger(__name__)

# Default retry configuration
DEFAULT_MAX_RETRIES = 10
DEFAULT_BASE_DELAY = 1.0  # seconds
DEFAULT_MAX_DELAY = 60.0  # seconds


class LLMRetryConfig:
    """Configuration for LLM retry behavior."""

    def __init__(self,
                 max_retries: int = DEFAULT_MAX_RETRIES,
                 base_delay: float = DEFAULT_BASE_DELAY,
                 max_delay: float = DEFAULT_MAX_DELAY,
                 allow_empty_response: bool = False):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.allow_empty_response = allow_empty_response


async def retry_llm_ainvoke(
    llm_call: Callable,
    messages: list[BaseMessage],
    config: RunnableConfig,
    retry_config: LLMRetryConfig | None = None,
    agent_prefix: str = "Agent"
) -> Any:
    """
    Retry wrapper for LLM ainvoke calls with exponential backoff and jitter.

    Args:
        llm_call: The LLM ainvoke method to call
        messages: The messages to pass to the LLM
        config: The runnable config
        retry_config: Optional retry configuration
        agent_prefix: Prefix for logging (e.g., "Tool Calling Agent")

    Returns:
        The LLM response

    Raises:
        The last exception if all retries are exhausted
    """
    if retry_config is None:
        retry_config = LLMRetryConfig()

    last_exception = None
    for attempt in range(retry_config.max_retries):
        try:
            response = await llm_call(messages, config=config)

            # Check for empty response
            if not retry_config.allow_empty_response:
                # A response is considered empty only if it has no content AND no tool calls
                has_content = response and response.content and response.content.strip() != ""
                has_tool_calls = hasattr(response, 'tool_calls') and response.tool_calls

                if not has_content and not has_tool_calls:
                    raise ValueError("LLM returned empty response (no content and no tool calls)")

            return response

        except Exception as e:
            last_exception = e
            logger.warning(
                "%s LLM invocation failed (attempt %d/%d): %s: %s",
                agent_prefix,
                attempt + 1,
                retry_config.max_retries,
                type(e).__name__,
                str(e)
            )

            # If this is the last attempt, raise the exception
            if attempt >= retry_config.max_retries - 1:
                logger.error("%s All %d retry attempts failed", agent_prefix, retry_config.max_retries)
                raise

            # Calculate delay with exponential backoff plus jitter
            delay = min(
                retry_config.base_delay * (2 ** attempt) + random.uniform(0, 1),
                retry_config.max_delay
            )
            logger.info("%s Retrying in %.1f seconds...", agent_prefix, delay)
            await asyncio.sleep(delay)

    # Just in case
    raise last_exception or RuntimeError("Unexpected: Exited retry loop without returning")


async def retry_llm_astream(
    llm_stream_call: Callable,
    input_data: dict,
    config: RunnableConfig,
    retry_config: LLMRetryConfig | None = None,
    agent_prefix: str = "Agent"
) -> str:
    """
    Retry wrapper for LLM astream calls with exponential backoff and jitter.

    Args:
        llm_stream_call: The LLM astream method to call
        input_data: The input dictionary to pass to the LLM
        config: The runnable config
        retry_config: Optional retry configuration
        agent_prefix: Prefix for logging (e.g., "ReAct Agent")

    Returns:
        The accumulated string response from the stream

    Raises:
        The last exception if all retries are exhausted
    """
    if retry_config is None:
        retry_config = LLMRetryConfig()

    last_exception = None

    for attempt in range(retry_config.max_retries):
        try:
            output_message = ""
            async for event in llm_stream_call(input_data, config=config):
                # Handle different event types - could be string or object with content
                if hasattr(event, 'content'):
                    output_message += event.content
                else:
                    output_message += str(event)

            # Check for empty response
            # Note: Tool calls are typically not available in streaming mode
            if not retry_config.allow_empty_response and output_message.strip() == "":
                raise ValueError("LLM returned empty response")

            return output_message

        except Exception as e:
            last_exception = e
            logger.warning(
                "%s LLM streaming invocation failed (attempt %d/%d): %s: %s",
                agent_prefix,
                attempt + 1,
                retry_config.max_retries,
                type(e).__name__,
                str(e)
            )

            # If this is the last attempt, raise the exception
            if attempt >= retry_config.max_retries - 1:
                logger.error("%s All %d retry attempts failed", agent_prefix, retry_config.max_retries)
                raise

            # Calculate delay with exponential backoff plus jitter
            delay = min(
                retry_config.base_delay * (2 ** attempt) + random.uniform(0, 1),
                retry_config.max_delay
            )
            logger.info("%s Retrying in %.1f seconds...", agent_prefix, delay)
            await asyncio.sleep(delay)

    # Just in case
    raise last_exception or RuntimeError("Unexpected: Exited retry loop without returning")