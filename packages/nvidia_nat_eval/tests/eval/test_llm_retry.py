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

import pytest

from nat.utils.exception_handlers.automatic_retries import patch_with_retry

DEFAULT_RETRY_STATUS_CODES = [429, 500, 502, 503, 504]
DEFAULT_RETRY_ERROR_MESSAGES = [
    "Too Many Requests",  # 429
    "429",  # 429 (numeric form)
    "Internal Server Error",  # 500
    "Bad Gateway",  # 502
    "Service Unavailable",  # 503
    "Gateway Timeout",  # 504
]
DEFAULT_NUM_RETRIES = 3


@pytest.mark.parametrize("error_msg", DEFAULT_RETRY_ERROR_MESSAGES)
async def test_evaluator_llm_retries_default_error_message(error_msg: str):
    """Evaluator LLM retries errors matching default retry_on_errors config."""
    call_count = 0

    class MockLLM:

        async def invoke(self, prompt: str) -> str:
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise Exception(f"Error: {error_msg}")
            return "Success"

    wrapped = patch_with_retry(
        MockLLM(),
        retries=DEFAULT_NUM_RETRIES,
        retry_codes=DEFAULT_RETRY_STATUS_CODES,
        retry_on_messages=DEFAULT_RETRY_ERROR_MESSAGES,
    )

    result = await wrapped.invoke("test")
    assert result == "Success"
    assert call_count == 2


async def test_evaluator_llm_retries_custom_error_message():
    """User-configured LLM error messages override defaults."""
    call_count = 0
    custom_error_messages = ["CustomRetryableError"]

    class MockLLM:

        async def invoke(self, prompt: str) -> str:
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise Exception("CustomRetryableError occurred")
            return "Success"

    wrapped = patch_with_retry(
        MockLLM(),
        retries=DEFAULT_NUM_RETRIES,
        retry_codes=[],
        retry_on_messages=custom_error_messages,
    )

    result = await wrapped.invoke("test")
    assert result == "Success"
    assert call_count == 2


async def test_evaluator_llm_custom_config_removes_defaults():
    """Custom LLM config removes default retry behavior."""
    call_count = 0

    class MockLLM:

        async def invoke(self, prompt: str) -> str:
            nonlocal call_count
            call_count += 1
            raise Exception("Too Many Requests")

    wrapped = patch_with_retry(
        MockLLM(),
        retries=DEFAULT_NUM_RETRIES,
        retry_codes=[],
        retry_on_messages=["CustomError"],
    )

    with pytest.raises(Exception, match="Too Many Requests"):
        await wrapped.invoke("test")

    assert call_count == 1
