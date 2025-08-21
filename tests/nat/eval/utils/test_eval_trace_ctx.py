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

from unittest.mock import MagicMock
from unittest.mock import patch

import pytest

from nat.eval.utils.eval_trace_ctx import EvalTraceContext
from nat.eval.utils.eval_trace_ctx import WeaveEvalTraceContext

# pylint: disable=redefined-outer-name


@pytest.fixture
def mock_eval_call():
    """Fixture to provide a mock evaluation call object."""
    call = MagicMock()
    call.id = "test-eval-call-123"
    return call


@pytest.fixture
def base_trace_context():
    """Fixture to provide a base EvalTraceContext instance."""
    return EvalTraceContext()


@pytest.fixture
def weave_trace_context():
    """Fixture to provide a WeaveEvalTraceContext instance."""
    return WeaveEvalTraceContext()


def test_eval_trace_context_set_eval_call_without_id(base_trace_context):
    """Test that set_eval_call handles objects without id attribute."""
    call_without_id = "simple-string-call"

    base_trace_context.set_eval_call(call_without_id)

    assert base_trace_context.eval_call == call_without_id


def test_weave_trace_context_init_weave_not_available():
    """Test WeaveEvalTraceContext initialization when Weave is not available."""
    with patch("builtins.__import__", side_effect=ImportError("No module named 'weave'")):
        ctx = WeaveEvalTraceContext()

        assert ctx.available is False
        assert ctx.set_call_stack is None


def test_weave_evaluation_context_with_weave(mock_eval_call):
    """Test WeaveEvalTraceContext evaluation_context when Weave is available."""
    mock_set_call_stack = MagicMock()
    mock_context_manager = MagicMock()
    mock_set_call_stack.return_value = mock_context_manager

    ctx = WeaveEvalTraceContext()
    ctx.available = True
    ctx.eval_call = mock_eval_call
    ctx.set_call_stack = mock_set_call_stack

    context_entered = False

    with ctx.evaluation_context():
        context_entered = True

    assert context_entered
    mock_set_call_stack.assert_called_once_with([mock_eval_call])


def test_weave_evaluation_context_without_weave(mock_eval_call):
    """Test WeaveEvalTraceContext evaluation_context when Weave is not available."""
    ctx = WeaveEvalTraceContext()
    ctx.available = False
    ctx.eval_call = mock_eval_call

    context_entered = False

    with ctx.evaluation_context():
        context_entered = True

    assert context_entered


def test_weave_evaluation_context_exception_handling(mock_eval_call):
    """Test WeaveEvalTraceContext evaluation_context handles exceptions gracefully."""
    # Create a mock set_call_stack that raises an exception when called
    mock_set_call_stack = MagicMock(side_effect=RuntimeError("Weave context failed"))

    ctx = WeaveEvalTraceContext()
    ctx.available = True
    ctx.eval_call = mock_eval_call
    ctx.set_call_stack = mock_set_call_stack

    context_entered = False

    with ctx.evaluation_context():
        context_entered = True

    assert context_entered
