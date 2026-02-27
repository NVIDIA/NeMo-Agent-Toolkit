# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use it except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Invariant tests for uncertainty principle: easy patterns -> full, difficult -> partial."""

import pytest

from nat_app.api import analyze_function

# --- Easy patterns (5): should get full confidence ---


def easy_dict_read_write(state):
    return {"result": state["query"]}


def easy_dict_get(state):
    return {"result": state.get("query", "")}


def easy_dict_keys(state):
    return {"keys": list(state.keys())}


def easy_dict_values(state):
    return {"vals": list(state.values())}


def easy_dict_items(state):
    return {"items": list(state.items())}


# --- Difficult patterns (16+): should get partial confidence ---


def difficult_exec(state):
    exec("x=1")  # noqa: S102
    return {}


def difficult_eval(state):
    eval("state")  # noqa: S307
    return {}


# Module-level var captured as closure freevar
_closure_outer = {}


def difficult_closure_write(state):
    _closure_outer["x"] = state.get("input", 1)
    return {}


def difficult_global_write(state):
    module_var["x"] = state.get("input", 1)  # noqa: F821
    return {}


def difficult_unknown_attr(state):
    x = state.some_attr  # noqa: F821
    return {"result": x}


def difficult_return_lambda_mutates(state):
    return lambda: state.update({"delayed": True})


def difficult_dynamic_attr(obj, attr, val):
    setattr(obj, attr, val)
    return {}


def difficult_state_custom_method(state):
    state.custom_helper()  # noqa: F821
    return {}


def difficult_dynamic_key(state):
    key = some_func()  # noqa: F821
    return {key: "val"}


def difficult_warnings_no_writes():
    return {}


def difficult_property_like_read(state):
    return {"x": state.some_property}  # noqa: F821


def difficult_compile(state):
    compile("x=1", "<string>", "exec")  # noqa: S102
    return {}


def difficult_getattr_dynamic(state, attr):
    return getattr(state, attr)


# --- Test classes ---


class TestEasyPatternsFullConfidence:
    """All easy patterns must get full confidence."""

    @pytest.mark.parametrize("fn", [
        easy_dict_read_write,
        easy_dict_get,
        easy_dict_keys,
        easy_dict_values,
        easy_dict_items,
    ])
    def test_easy_pattern_full_confidence(self, fn):
        info = analyze_function(fn)
        assert info["confidence"] == "full", (f"{fn.__name__} expected full confidence, got {info['confidence']}")


class TestDifficultPatternsPartialConfidence:
    """All difficult patterns must get partial (or opaque) confidence."""

    @pytest.mark.parametrize("fn",
                             [
                                 difficult_exec,
                                 difficult_eval,
                                 difficult_closure_write,
                                 difficult_global_write,
                                 difficult_unknown_attr,
                                 difficult_return_lambda_mutates,
                                 difficult_dynamic_attr,
                                 difficult_state_custom_method,
                                 difficult_dynamic_key,
                                 difficult_warnings_no_writes,
                                 difficult_property_like_read,
                                 difficult_compile,
                                 difficult_getattr_dynamic,
                             ])
    def test_difficult_pattern_partial_confidence(self, fn):
        info = analyze_function(fn)
        assert info["confidence"] in ("partial", "opaque"), (
            f"{fn.__name__} expected partial/opaque confidence, got {info['confidence']}")
