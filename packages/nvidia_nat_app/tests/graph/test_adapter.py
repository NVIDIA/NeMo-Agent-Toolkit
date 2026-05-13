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
"""Tests for AbstractFrameworkAdapter: abstract methods, defaults, and analyze_node."""

import pytest

from nat_app.graph.adapter import AbstractFrameworkAdapter


class _MinimalAdapter(AbstractFrameworkAdapter):

    def extract(self, source):
        from nat_app.graph.types import Graph

        return Graph()

    def build(self, original, result):
        return original


class TestAbstractMethods:

    def test_cannot_instantiate_base_class(self):
        with pytest.raises(TypeError, match="abstract"):
            AbstractFrameworkAdapter()

    def test_subclass_missing_extract_cannot_instantiate(self):

        class IncompleteAdapter(AbstractFrameworkAdapter):

            def build(self, original, result):
                return original

        with pytest.raises(TypeError, match="abstract"):
            IncompleteAdapter()

    def test_subclass_missing_build_cannot_instantiate(self):

        class IncompleteAdapter(AbstractFrameworkAdapter):

            def extract(self, source):
                from nat_app.graph.types import Graph

                return Graph()

        with pytest.raises(TypeError, match="abstract"):
            IncompleteAdapter()


class TestDefaults:

    @pytest.mark.parametrize(
        "method, args, expected",
        [
            ("get_node_func", ("any_node", ), None),
            ("get_state_schema", (), None),
            ("get_reducer_fields", (), {}),
            ("get_all_schema_fields", (), None),
            ("get_special_call_names", (), set()),
            ("get_param_to_obj", (), None),
            ("get_self_state_attrs", (), None),
            ("get_llm_detector", (), None),
        ],
        ids=lambda v: v if isinstance(v, str) else "",
    )
    def test_default_return_values(self, method, args, expected):
        adapter = _MinimalAdapter()
        assert getattr(adapter, method)(*args) == expected

    def test_map_profiler_function_default(self):
        adapter = _MinimalAdapter()
        assert adapter.map_profiler_function_to_node("my_func") == "my_func"


class TestAnalyzeNode:

    def test_source_available(self):
        adapter = _MinimalAdapter()

        def my_func(state):
            return {"result": state["query"]}

        analysis = adapter.analyze_node("test", my_func)
        assert analysis.name == "test"
        assert analysis.source == "ast"
        assert analysis.confidence == "full"
        assert "query" in analysis.reads.all_fields_flat

    def test_source_unavailable(self):
        adapter = _MinimalAdapter()
        analysis = adapter.analyze_node("test", len)
        assert analysis.confidence == "opaque"
        assert analysis.source == "unavailable"

    def test_schema_fallback_on_opaque(self):
        adapter = _MinimalAdapter()
        analysis = adapter.analyze_node("test", len, all_schema_fields={"a", "b"})
        assert analysis.confidence == "opaque"
        assert analysis.mutations.all_fields_flat == {"a", "b"}

    def test_warnings_aggregated(self):
        adapter = _MinimalAdapter()
        analysis = adapter.analyze_node("test", len)
        assert len(analysis.warnings) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
