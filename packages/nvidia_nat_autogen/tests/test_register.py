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
"""Test register.py file"""

import pytest


class TestRegisterModule:
    """Test cases for register module."""

    def test_imports_exist(self):
        """Test that required imports exist and work."""
        try:
            from nat.plugins.autogen import llm  # pylint: disable=import-outside-toplevel
            from nat.plugins.autogen import tool_wrapper  # pylint: disable=import-outside-toplevel
            assert llm is not None
            assert tool_wrapper is not None
        except ImportError as e:
            pytest.fail(f"Failed to import required modules: {e}")

    def test_register_module_importable(self):
        """Test that the register module can be imported."""
        try:
            from nat.plugins.autogen import register  # pylint: disable=import-outside-toplevel
            assert register is not None
        except ImportError as e:
            pytest.fail(f"Failed to import register module: {e}")

    def test_llm_module_functions(self):
        """Test that LLM module has expected functions."""
        from nat.plugins.autogen import llm  # pylint: disable=import-outside-toplevel

        # Check for expected functions
        expected_functions = ['openai_autogen', 'azure_openai_autogen', 'nim_autogen']

        for func_name in expected_functions:
            assert hasattr(llm, func_name), f"Function {func_name} not found in llm module"

    def test_tool_wrapper_module_functions(self):
        """Test that tool_wrapper module has expected functions."""
        from nat.plugins.autogen import tool_wrapper  # pylint: disable=import-outside-toplevel

        # Check for expected functions
        expected_functions = ['resolve_type', 'autogen_tool_wrapper']

        for func_name in expected_functions:
            assert hasattr(tool_wrapper, func_name), f"Function {func_name} not found in tool_wrapper module"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
