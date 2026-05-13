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
"""Tests for constraint decorators: @sequential, @depends_on, @has_side_effects."""

import pytest

from nat_app.constraints.decorators import depends_on
from nat_app.constraints.decorators import has_side_effects
from nat_app.constraints.decorators import sequential
from nat_app.constraints.models import NodeConstraints


class TestSequential:

    def test_sets_force_sequential(self):

        @sequential(reason="DB write")
        def my_node(state):
            pass

        constraints = my_node._optimization_constraints
        assert constraints.force_sequential is True

    def test_sets_has_side_effects(self):

        @sequential(reason="DB write")
        def my_node(state):
            pass

        assert my_node._optimization_constraints.has_side_effects is True

    def test_stores_reason(self):

        @sequential(reason="DB write")
        def my_node(state):
            pass

        assert my_node._optimization_constraints.reason == "DB write"

    def test_default_reason_when_omitted(self):

        @sequential()
        def my_node(state):
            pass

        assert my_node._optimization_constraints.reason == "Marked as sequential"

    def test_returns_original_function(self):

        @sequential(reason="test")
        def my_node(state):
            return "result"

        assert my_node(None) == "result"

    def test_constraints_type(self):

        @sequential(reason="test")
        def my_node(state):
            pass

        assert isinstance(my_node._optimization_constraints, NodeConstraints)


class TestDependsOn:

    def test_populates_depends_on_set(self):

        @depends_on("fetch_data", "validate_input")
        def my_node(state):
            pass

        constraints = my_node._optimization_constraints
        assert constraints.depends_on == {"fetch_data", "validate_input"}

    def test_single_dependency(self):

        @depends_on("upstream")
        def my_node(state):
            pass

        assert my_node._optimization_constraints.depends_on == {"upstream"}

    def test_stores_reason(self):

        @depends_on("a", reason="Needs A complete")
        def my_node(state):
            pass

        assert my_node._optimization_constraints.reason == "Needs A complete"

    def test_no_reason_leaves_none(self):

        @depends_on("a")
        def my_node(state):
            pass

        assert my_node._optimization_constraints.reason is None

    def test_does_not_set_force_sequential(self):

        @depends_on("a")
        def my_node(state):
            pass

        assert my_node._optimization_constraints.force_sequential is False

    def test_returns_original_function(self):

        @depends_on("a")
        def my_node(state):
            return 42

        assert my_node(None) == 42


class TestHasSideEffects:

    def test_sets_has_side_effects(self):

        @has_side_effects(reason="HTTP call")
        def my_node(state):
            pass

        assert my_node._optimization_constraints.has_side_effects is True

    def test_does_not_set_force_sequential(self):

        @has_side_effects(reason="HTTP call")
        def my_node(state):
            pass

        assert my_node._optimization_constraints.force_sequential is False

    def test_stores_reason(self):

        @has_side_effects(reason="Sends email")
        def my_node(state):
            pass

        assert my_node._optimization_constraints.reason == "Sends email"

    def test_default_reason_when_omitted(self):

        @has_side_effects()
        def my_node(state):
            pass

        assert my_node._optimization_constraints.reason == "Has side effects"

    def test_returns_original_function(self):

        @has_side_effects(reason="test")
        def my_node(state):
            return "ok"

        assert my_node(None) == "ok"


class TestDecoratorStacking:

    def test_sequential_plus_depends_on(self):

        @sequential(reason="Must be ordered")
        @depends_on("upstream")
        def my_node(state):
            pass

        constraints = my_node._optimization_constraints
        assert constraints.force_sequential is True
        assert "upstream" in constraints.depends_on

    def test_has_side_effects_plus_depends_on(self):

        @has_side_effects(reason="API call")
        @depends_on("auth", "validate")
        def my_node(state):
            pass

        constraints = my_node._optimization_constraints
        assert constraints.has_side_effects is True
        assert constraints.depends_on == {"auth", "validate"}
        assert constraints.force_sequential is False

    def test_multiple_depends_on_accumulate(self):

        @depends_on("c")
        @depends_on("a", "b")
        def my_node(state):
            pass

        constraints = my_node._optimization_constraints
        assert constraints.depends_on == {"a", "b", "c"}


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
