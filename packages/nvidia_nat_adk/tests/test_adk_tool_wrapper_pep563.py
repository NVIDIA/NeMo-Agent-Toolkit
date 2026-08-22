# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Regression test for PEP 563 annotations in ADK tool wrapper (issue #2161)."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock
from unittest.mock import patch

from pydantic import BaseModel

from nat.plugins.adk.tool_wrapper import google_adk_tool_wrapper

# ----------------------------
# PEP 563 Models for Testing
# ----------------------------


class Pep563Input(BaseModel):
    """Input model for PEP 563 test function.

    Attributes:
        text: The text to echo.
        count: The number of times to repeat the text.
    """

    text: str
    count: int


class Pep563Output(BaseModel):
    """Output model for PEP 563 test function.

    Attributes:
        result: The echoed text repeated count times.
    """

    result: str


class Pep563Function:
    """Function using PEP 563 annotations for testing.

    Attributes:
        description: Human-readable description of the function.
        config: Configuration object with function type.
        has_single_output: Whether the function has a single output.
        has_streaming_output: Whether the function streams output.
        input_schema: Pydantic model for input validation.
        single_output_schema: Pydantic model for single output validation.
        streaming_output_schema: Pydantic model for streaming output validation (None here).
    """

    def __init__(self) -> None:
        self.description = "PEP 563 ADK function"
        self.config = type('Config', (), {'type': 'pep563_adk_func'})
        self.has_single_output = True
        self.has_streaming_output = False
        self.input_schema = Pep563Input
        self.single_output_schema = Pep563Output
        self.streaming_output_schema = None

    async def acall_invoke(self, *args: Any, **_kwargs: Any) -> Pep563Output:
        """Invoke the function with the given arguments.

        Args:
            *args: Positional arguments, expects first arg to be Pep563Input.
            **_kwargs: Keyword arguments (not used).

        Returns:
            Pep563Output: The function result.
        """
        input_obj = args[0]
        return Pep563Output(result=f"{input_obj.text} x {input_obj.count}")


# ----------------------------
# Pytest Unit Tests
# ----------------------------


@patch('google.adk.tools.function_tool.FunctionTool')
def test_google_adk_tool_wrapper_pep563_annotations(mock_function_tool: MagicMock) -> None:
    """Test the ADK tool wrapper with PEP 563 annotations.

    This is a regression test for issue #2161 where string annotations from
    PEP 563 caused KeyError in ADK's deferred annotation resolution.
    """
    dummy_fn = Pep563Function()
    mock_builder = MagicMock()

    mock_tool_instance = MagicMock()
    mock_function_tool.return_value = mock_tool_instance

    # Call the wrapper - this should not raise KeyError
    result = google_adk_tool_wrapper('pep563_adk_func', dummy_fn, mock_builder)

    # Verify FunctionTool was called
    assert mock_function_tool.called
    assert result == mock_tool_instance

    # Verify the callable was created with correct metadata
    call_args = mock_function_tool.call_args[0][0]
    assert call_args.__name__ == 'pep563_adk_func'
    assert call_args.__doc__ == "PEP 563 ADK function"

    # Verify signature has correct types (not strings)
    sig = call_args.__signature__
    assert sig is not None
    params = sig.parameters
    assert 'text' in params
    assert 'count' in params
    # The annotations should be actual types, not strings
    assert params['text'].annotation is str
    assert params['count'].annotation is int
