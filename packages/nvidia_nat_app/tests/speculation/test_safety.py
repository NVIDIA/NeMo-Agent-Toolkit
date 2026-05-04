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
"""Tests for speculation safety: @speculation_unsafe, is_marked_unsafe, SpeculationSafetyConfig, RouterDescriptor."""

import pytest

from nat_app.speculation.safety import RouterDescriptor
from nat_app.speculation.safety import SpeculationSafetyConfig
from nat_app.speculation.safety import is_marked_speculation_unsafe
from nat_app.speculation.safety import speculation_unsafe


class TestSpeculationUnsafeDecorator:

    def test_marks_function(self):

        @speculation_unsafe
        def my_func(state):
            pass

        assert my_func._speculation_unsafe is True

    def test_marks_class(self):

        @speculation_unsafe
        class MyMiddleware:
            pass

        assert MyMiddleware._speculation_unsafe is True

    def test_returns_original_function(self):

        @speculation_unsafe
        def my_func(state):
            return "result"

        assert my_func(None) == "result"

    def test_returns_original_class(self):

        @speculation_unsafe
        class MyClass:
            value = 42

        assert MyClass.value == 42


class TestIsMarkedSpeculationUnsafe:

    def test_true_for_decorated_function(self):

        @speculation_unsafe
        def unsafe_fn(state):
            pass

        assert is_marked_speculation_unsafe(unsafe_fn) is True

    def test_true_for_decorated_class(self):

        @speculation_unsafe
        class UnsafeClass:
            pass

        assert is_marked_speculation_unsafe(UnsafeClass) is True

    def test_false_for_plain_function(self):

        def safe_fn(state):
            pass

        assert is_marked_speculation_unsafe(safe_fn) is False

    def test_false_for_plain_class(self):

        class SafeClass:
            pass

        assert is_marked_speculation_unsafe(SafeClass) is False

    def test_false_for_none(self):
        assert is_marked_speculation_unsafe(None) is False

    def test_false_for_arbitrary_object(self):
        assert is_marked_speculation_unsafe(42) is False


class TestSpeculationSafetyConfig:

    def test_defaults(self):
        config = SpeculationSafetyConfig()
        assert config.unsafe_nodes == set()
        assert config.safe_overrides == set()

    def test_unsafe_nodes(self):
        config = SpeculationSafetyConfig(unsafe_nodes={"a", "b"})
        assert config.unsafe_nodes == {"a", "b"}

    def test_safe_overrides(self):
        config = SpeculationSafetyConfig(safe_overrides={"x"})
        assert config.safe_overrides == {"x"}

    def test_instances_do_not_share_sets(self):
        c1 = SpeculationSafetyConfig()
        c2 = SpeculationSafetyConfig()
        c1.unsafe_nodes.add("node_x")
        assert "node_x" not in c2.unsafe_nodes


class TestRouterDescriptor:

    def test_fields(self):
        rd = RouterDescriptor(name="my_router", possible_targets=["a", "b"])
        assert rd.name == "my_router"
        assert rd.possible_targets == ["a", "b"]

    def test_decision_fn_defaults_none(self):
        rd = RouterDescriptor(name="r", possible_targets=[])
        assert rd.decision_fn is None

    def test_decision_fn_callable(self):

        def choose(state):
            return "a"

        rd = RouterDescriptor(name="r", possible_targets=["a", "b"], decision_fn=choose)
        assert rd.decision_fn is choose
        assert rd.decision_fn({"x": 1}) == "a"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
