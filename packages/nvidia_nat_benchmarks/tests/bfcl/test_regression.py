# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Regression tests for BFCL evaluator — pinned behavior for known outputs."""

import pytest

from nat.plugins.benchmarks.bfcl.evaluator import _extract_function_call


class TestExtractFunctionCall:

    def test_bare_function_call(self):
        assert _extract_function_call("calc(x=10)") == "calc(x=10)"

    def test_bracketed_function_call(self):
        assert _extract_function_call("[calc(x=10)]") == "[calc(x=10)]"

    def test_markdown_code_block(self):
        raw = """Here is the function call:

```python
calculate_area(base=10, height=5)
```"""
        assert _extract_function_call(raw) == "calculate_area(base=10, height=5)"

    def test_markdown_tool_code_block(self):
        raw = """```tool_code
func(a=1, b=2)
```"""
        assert _extract_function_call(raw) == "func(a=1, b=2)"

    def test_prose_with_function_call_line(self):
        raw = """The function to call is:
calculate_area(base=10, height=5)
This will compute the area."""
        assert _extract_function_call(raw) == "calculate_area(base=10, height=5)"

    def test_tools_prefix_stripped(self):
        raw = "tools.calculate_area(base=10, height=5)"
        result = _extract_function_call(raw)
        assert "calculate_area(base=10, height=5)" in result

    def test_multiple_calls_extracted(self):
        raw = """func_a(x=1)
func_b(y=2)"""
        result = _extract_function_call(raw)
        assert "func_a(x=1)" in result
        assert "func_b(y=2)" in result

    def test_no_function_call_returns_raw(self):
        raw = "I cannot help with that request."
        assert _extract_function_call(raw) == raw
