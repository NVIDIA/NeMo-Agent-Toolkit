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
"""Tests for AST-based static analysis of node functions."""

import pytest

from nat_app.graph.static_analysis import StaticAnalysisResult
from nat_app.graph.static_analysis import analyze_function_ast


class TestAnalyzeFunctionAST:

    def test_dict_read_via_subscript(self):

        def fn(state):
            x = state["query"]
            return {"result": x}

        r = analyze_function_ast(fn)
        assert "query" in r.reads.all_fields_flat

    def test_dict_read_via_get(self):

        def fn(state):
            x = state.get("query")
            return {"result": x}

        r = analyze_function_ast(fn)
        assert "query" in r.reads.all_fields_flat

    def test_dict_write_via_return(self):

        def fn(_state):
            return {"result": "done"}

        r = analyze_function_ast(fn)
        assert "result" in r.writes.all_fields_flat

    def test_mutating_method_append(self):

        def fn(state):
            state["messages"].append("hi")
            return {}

        r = analyze_function_ast(fn)
        assert "messages" in r.mutations.all_fields_flat

    def test_mutating_method_update(self):

        def fn(state):
            state["data"].update({"key": "val"})
            return {}

        r = analyze_function_ast(fn)
        assert "data" in r.mutations.all_fields_flat

    def test_source_unavailable_for_builtin(self):
        r = analyze_function_ast(len)
        assert not r.source_available

    def test_dynamic_key_flagged(self):

        def fn(_state):
            key = some_func()  # noqa: F821
            return {key: "val"}

        r = analyze_function_ast(fn)
        assert r.has_dynamic_keys

    def test_special_call_detection(self):

        def fn(state):
            Send("target", {"data": state["x"]})  # noqa: F821
            return {}

        r = analyze_function_ast(fn, special_call_names={"Send"})
        assert "Send" in r.detected_special_calls

    def test_lambda_analysis(self):
        fn = lambda state: {"output": state["input"]}  # noqa: E731
        r = analyze_function_ast(fn)
        assert "output" in r.writes.all_fields_flat
        assert "input" in r.reads.all_fields_flat

    def test_starred_unpacking_no_crash(self):
        """Starred unpacking (a, *rest = x, y, z) has different lengths; must not crash."""

        def fn(state):
            a, *_ = state["x"], state["y"], state["z"]
            return {"out": a}

        r = analyze_function_ast(fn)
        assert r.source_available
        assert "x" in r.reads.all_fields_flat or "y" in r.reads.all_fields_flat
        assert "out" in r.writes.all_fields_flat

    def test_confidence_full_simple(self):

        def fn(state):
            return {"result": state["query"]}

        r = analyze_function_ast(fn)
        assert not r.has_dynamic_keys
        assert not r.has_unresolved_calls

    def test_self_state_attrs(self):

        class MyFlow:

            def step(self):
                x = self.state["query"]
                self.state["result"] = x

        r = analyze_function_ast(MyFlow.step, self_state_attrs={"state": "state"})
        assert "query" in r.reads.all_fields_flat
        assert "result" in r.mutations.all_fields_flat

    def test_augassign_detected(self):

        def fn(state):
            state["count"] += 1
            return {}

        r = analyze_function_ast(fn)
        assert "count" in r.mutations.all_fields_flat
        assert "count" in r.reads.all_fields_flat

    def test_no_params_warning(self):

        def fn():
            return {}

        r = analyze_function_ast(fn)
        assert any("no parameters" in w.lower() or "no param" in w.lower() for w in r.warnings)

    def test_async_function(self):

        async def fn(state):
            return {"result": state["query"]}

        r = analyze_function_ast(fn)
        assert "query" in r.reads.all_fields_flat
        assert "result" in r.writes.all_fields_flat


class TestDelete:

    def test_del_state_subscript(self):

        def fn(state):
            del state["key"]
            return {}

        r = analyze_function_ast(fn)
        assert "key" in r.mutations.all_fields_flat

    def test_del_closure_subscript(self):
        outer = {}

        def fn(_state):
            del outer["x"]
            return {}

        r = analyze_function_ast(fn)
        assert r.has_closure_write


class TestReturnDictSpread:

    def test_return_dict_spread(self):

        def fn(state):
            v = state.get("old", "")
            return {**state, "new": v}

        r = analyze_function_ast(fn)
        assert r.has_dynamic_keys

    def test_return_dict_non_literal_key(self):

        def fn(state):
            key = some_func()  # noqa: F821
            val = state.get("x")
            return {key: val}

        r = analyze_function_ast(fn)
        assert r.has_dynamic_keys


class TestStatePassedToCall:

    def test_state_passed_to_unresolved(self):

        def fn(state):
            helper(state)  # noqa: F821
            return {}

        r = analyze_function_ast(fn)
        assert r.has_unresolved_calls

    def test_state_index_passed_to_unresolved(self):

        def fn(state):
            helper(state["x"])  # noqa: F821
            return {}

        r = analyze_function_ast(fn)
        # state["x"] is passed to helper; analyzer tracks the read
        assert "x" in r.reads.all_fields_flat


class TestKnownSafeAttrs:

    def test_state_copy_full_confidence(self):

        def fn(state):
            return state.copy()

        r = analyze_function_ast(fn)
        assert not r.has_unknown_attr_access
        assert not r.has_dynamic_exec
        assert not r.has_closure_write
        assert not r.has_global_write

    def test_state_keys_full_confidence(self):

        def fn(state):
            return {"keys": list(state.keys())}

        r = analyze_function_ast(fn)
        assert not r.has_unknown_attr_access


class TestParamToObj:

    def test_multi_param_tracks_both(self):

        def fn(state, memory):
            x = state["query"]
            memory["cache"] = x
            return {}

        r = analyze_function_ast(fn, param_to_obj={"state": "state", "memory": "memory"})
        assert "query" in r.reads.all_fields_flat
        assert "cache" in r.mutations.all_fields_flat or "cache" in r.writes.all_fields_flat

    def test_empty_param_to_obj_raises(self):
        """Empty param_to_obj raises ValueError instead of StopIteration."""

        def fn(state):
            return state.get("x", {})

        with pytest.raises(ValueError, match="param_to_obj must contain at least one mapping"):
            analyze_function_ast(fn, param_to_obj={})

    def test_vararg_state_access(self):

        def fn(*args):
            if args:
                args[0]["x"] = 1
            return {}

        r = analyze_function_ast(fn)
        # args[0] has numeric index; analyzer treats as dynamic (conservative)
        assert r.has_dynamic_keys or "x" in r.mutations.all_fields_flat or "x" in r.writes.all_fields_flat


class TestRecursionDepth:

    def test_recursion_depth_hit(self):
        """With default max_recursion_depth=5, a 7-level chain hits the limit."""

        def level6(s):
            return s

        def level5(s):
            return level6(s)

        def level4(s):
            return level5(s)

        def level3(s):
            return level4(s)

        def level2(s):
            return level3(s)

        def level1(s):
            return level2(s)

        def level0(s):
            return level1(s)

        r = analyze_function_ast(level0)
        assert r.recursion_depth_hit

    def test_recursion_depth_configurable(self):
        """max_recursion_depth=3 hits limit when level3 tries to call level4."""

        def level4(s):
            return s

        def level3(s):
            return level4(s)

        def level2(s):
            return level3(s)

        def level1(s):
            return level2(s)

        def level0(s):
            return level1(s)

        r = analyze_function_ast(level0, max_recursion_depth=3)
        assert r.recursion_depth_hit


class TestStaticAnalysisResult:

    def test_all_writes_combines(self):
        r = StaticAnalysisResult()
        r.writes.add("state", "a")
        r.mutations.add("state", "b")
        combined = r.all_writes
        flat = combined.all_fields_flat
        assert "a" in flat
        assert "b" in flat

    def test_defaults(self):
        r = StaticAnalysisResult()
        assert r.source_available is True
        assert r.has_dynamic_keys is False
        assert r.has_unresolved_calls is False
        assert r.has_dynamic_exec is False
        assert r.has_closure_write is False
        assert r.has_global_write is False
        assert r.has_unknown_attr_access is False
        assert r.has_return_lambda_mutates_state is False
        assert r.has_dynamic_attr is False
        assert r.warnings == []


class TestUncertaintyFlags:

    def test_has_dynamic_exec(self):

        def fn(_state):
            exec("x=1")  # noqa: S102
            return {}

        r = analyze_function_ast(fn)
        assert r.has_dynamic_exec

    def test_has_dynamic_exec_eval(self):

        def fn(_state):
            eval("state")  # noqa: S307
            return {}

        r = analyze_function_ast(fn)
        assert r.has_dynamic_exec

    def test_has_closure_write(self):
        outer = {}

        def fn(state):
            outer["x"] = state.get("input", 1)
            return {}

        r = analyze_function_ast(fn)
        assert r.has_closure_write

    def test_has_global_write(self):

        def fn(state):
            module_cache["x"] = state.get("input", 1)  # noqa: F821
            return {}

        r = analyze_function_ast(fn)
        assert r.has_global_write

    def test_has_unknown_attr_access(self):

        def fn(state):
            x = state.some_attr  # noqa: F821
            return {"result": x}

        r = analyze_function_ast(fn)
        assert r.has_unknown_attr_access

    def test_has_return_lambda_mutates_state(self):

        def fn(state):
            return lambda: state.update({"delayed": True})

        r = analyze_function_ast(fn)
        assert r.has_return_lambda_mutates_state

    def test_has_dynamic_attr(self):

        def fn(obj, attr, val):
            setattr(obj, attr, val)
            return {}

        r = analyze_function_ast(fn)
        assert r.has_dynamic_attr

    def test_state_as_receiver_unresolved(self):

        def fn(state):
            state.custom_helper()  # noqa: F821
            return {}

        r = analyze_function_ast(fn)
        assert r.has_unresolved_calls

    def test_augassign_closure_write(self):
        outer = {}

        def fn(_state):
            outer["x"] += 1
            return {}

        r = analyze_function_ast(fn)
        assert r.has_closure_write

    def test_augassign_global_write(self):

        def fn(_state):
            module_var["x"] += 1  # noqa: F821
            return {}

        r = analyze_function_ast(fn)
        assert r.has_global_write

    def test_has_dynamic_exec_compile(self):

        def fn(_state):
            compile("x=1", "<string>", "exec")  # noqa: S102
            return {}

        r = analyze_function_ast(fn)
        assert r.has_dynamic_exec


class TestChainedSubscript:

    def test_chained_subscript_read(self):

        def fn(state):
            x = state["a"]["b"]["c"]
            return {"result": x}

        r = analyze_function_ast(fn)
        assert "a" in r.reads.all_fields_flat or "a.b.c" in r.reads.all_fields_flat

    def test_chained_subscript_write(self):

        def fn(state):
            state["a"]["b"] = 1
            return {}

        r = analyze_function_ast(fn)
        assert "a" in r.mutations.all_fields_flat or "a.b" in r.mutations.all_fields_flat


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
