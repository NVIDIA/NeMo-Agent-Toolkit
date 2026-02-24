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
"""Tests for ResultHandler: should_merge dispatch and log_result."""

import pytest

from nat_app.executors.result_handler import ResultHandler


class TestShouldMerge:

    @pytest.fixture(name="handler")
    def fixture_handler(self):
        return ResultHandler()

    def test_none_result(self, handler):
        merge, desc = handler.should_merge(None)
        assert merge is False
        assert desc == "None"

    def test_dict_result(self, handler):
        merge, desc = handler.should_merge({"key": "val"})
        assert merge is True
        assert desc == "dict"

    def test_empty_dict(self, handler):
        merge, desc = handler.should_merge({})
        assert merge is True
        assert desc == "dict"

    def test_list_result(self, handler):
        merge, desc = handler.should_merge([1, 2, 3])
        assert merge is True
        assert desc == "list"

    def test_callable_result(self, handler):
        merge, desc = handler.should_merge(lambda x: x)
        assert merge is False
        assert desc.startswith("callable:")

    def test_unknown_type(self, handler):
        merge, desc = handler.should_merge(42)
        assert merge is False
        assert desc.startswith("unknown:")

    def test_string_is_unknown(self, handler):
        merge, desc = handler.should_merge("hello")
        assert merge is False
        assert desc.startswith("unknown:")


class TestCustomCommandChecker:

    def test_command_object_detected(self):

        class MyCommand:
            pass

        handler = ResultHandler(command_checker=lambda r: isinstance(r, MyCommand))
        merge, desc = handler.should_merge(MyCommand())
        assert merge is True
        assert desc.startswith("command:")

    def test_non_command_not_affected(self):
        handler = ResultHandler(command_checker=lambda r: False)
        merge, desc = handler.should_merge(42)
        assert merge is False
        assert desc.startswith("unknown:")


class TestLogResult:

    @pytest.fixture(name="handler")
    def fixture_handler(self):
        return ResultHandler()

    def test_log_none(self, handler):
        handler.log_result("node", None, False, "None")

    def test_log_dict(self, handler):
        handler.log_result("node", {"k": "v"}, True, "dict")

    def test_log_list(self, handler):
        handler.log_result("node", [1, 2], True, "list")

    def test_log_callable(self, handler):
        handler.log_result("node", lambda: None, False, "callable:function")

    def test_log_command(self, handler):
        handler.log_result("node", object(), True, "command:MyCommand")

    def test_log_unknown(self, handler):
        handler.log_result("node", 42, False, "unknown:int")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
