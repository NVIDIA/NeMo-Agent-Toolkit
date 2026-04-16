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
"""Regression tests for the chat_completion error-path sanitization (CWE-209).

The error handler in `nat.tool.chat_completion` must NOT surface any part of
the caught exception (message, stack frames, class names, file paths) in the
caller-visible response. These tests force the exception branch via a mocked
LLM and assert on the returned string.
"""

from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest

from nat.data_models.api_server import ChatRequest
from nat.data_models.api_server import Message
from nat.tool.chat_completion import ChatCompletionConfig
from nat.tool.chat_completion import register_chat_completion


# A distinctive, unlikely-to-appear-anywhere-else string so any substring match
# is definitive: either the exception leaks or it doesn't.
_SENTINEL = "UNIQUE-LEAK-SENTINEL-8c3b9fba"


async def _get_registered_callable(failing_llm: AsyncMock):
    """Drive the async generator register_chat_completion returns and pull
    the registered callable out of the yielded FunctionInfo.

    If the callable cannot be located, close the async generator before
    raising so the fixture doesn't leak an open generator.
    """
    config = ChatCompletionConfig(llm_name="test_llm")  # type: ignore[arg-type]

    builder = MagicMock()
    builder.get_llm = AsyncMock(return_value=failing_llm)

    gen = register_chat_completion(config, builder)
    fn_info = await gen.__anext__()
    # FunctionInfo wraps the inner function; pull it back out regardless of
    # which attribute name the current implementation uses.
    for attr in ("single_fn", "fn", "func", "_fn"):
        inner = getattr(fn_info, attr, None)
        if inner is not None and callable(inner):
            return inner, gen
    await gen.aclose()
    raise RuntimeError("could not locate the registered callable on FunctionInfo")


@pytest.fixture(name="failing_llm_runtime_error")
async def fixture_failing_llm_runtime_error():
    """Mocked chat completion wired to an LLM that raises RuntimeError.

    Yields (fn, llm) where `fn` is the registered callable and `llm` is the
    mocked failing LLM. Closes the underlying async generator on teardown so
    there's no resource leak across tests.
    """
    llm = AsyncMock()
    llm.ainvoke.side_effect = RuntimeError(_SENTINEL)
    fn, gen = await _get_registered_callable(llm)
    try:
        yield fn, llm
    finally:
        await gen.aclose()


@pytest.fixture(name="failing_llm_value_error")
async def fixture_failing_llm_value_error():
    """Same as above but the LLM raises ValueError with an embedded sentinel."""
    llm = AsyncMock()
    llm.ainvoke.side_effect = ValueError(f"boom {_SENTINEL}")
    fn, gen = await _get_registered_callable(llm)
    try:
        yield fn, llm
    finally:
        await gen.aclose()


class TestChatCompletionErrorSanitization:
    """The error response must never contain any part of the caught exception."""

    async def test_error_response_drops_exception_message(self, failing_llm_runtime_error):
        """LLM raises → response omits the exception text entirely."""
        fn, _llm = failing_llm_runtime_error
        request = ChatRequest(messages=[Message(role="user", content="hello there")])
        result = await fn(request)

        # Result may be str or ChatResponse depending on the `is_string` branch —
        # the ChatRequest path returns ChatResponse; coerce to string for the
        # leak check so we cover every sub-path.
        text = result if isinstance(result, str) else str(result)
        assert _SENTINEL not in text
        # And the RuntimeError class name must not appear either.
        assert "RuntimeError" not in text
        # The user-safe apology is what callers should see.
        assert "I apologize" in text

    async def test_error_response_echoes_user_query_but_not_exception(self, failing_llm_value_error):
        """Response should include the user's last message but not the exception."""
        fn, _llm = failing_llm_value_error
        request = ChatRequest(messages=[Message(role="user", content="what is my balance?")])
        result = await fn(request)

        text = result if isinstance(result, str) else str(result)
        assert "what is my balance?" in text  # the user's query is echoed
        assert _SENTINEL not in text  # but the exception text is not
        assert "ValueError" not in text

    async def test_server_side_logger_still_captures_full_exception(self, failing_llm_runtime_error):
        """Operators must still see the traceback in logs for triage."""
        fn, _llm = failing_llm_runtime_error
        request = ChatRequest(messages=[Message(role="user", content="test")])
        with patch("nat.tool.chat_completion.logger") as mock_logger:
            await fn(request)
            # logger.exception is the required call — it records the
            # traceback AND the message at ERROR level.
            mock_logger.exception.assert_any_call("chat completion failed")
