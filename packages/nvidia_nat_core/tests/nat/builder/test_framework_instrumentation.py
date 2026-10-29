# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import ast
import sys
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from nat.builder import framework_instrumentation
from nat.builder.framework_enum import LLMFrameworkEnum


def test_core_hook_imports_no_framework():
    """Framework packages register their instrumentation; the core module imports none of them."""
    source = framework_instrumentation.__file__
    with open(source, encoding="utf-8") as handle:
        tree = ast.parse(handle.read())

    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)

    allowed = {name for name in imported if name.split(".")[0] in sys.stdlib_module_names or name == "__future__"}
    allowed |= {name for name in imported if name.startswith(("nat.builder.", "nat.cli."))}
    assert imported == allowed, sorted(imported - allowed)


def test_register_decorator_registers_and_returns_the_function(monkeypatch):
    monkeypatch.setattr(framework_instrumentation, "_framework_instrumentation", {})

    @framework_instrumentation.register_framework_instrumentation(LLMFrameworkEnum.ADK)
    def instrument(workflow_llms):
        pass

    assert framework_instrumentation._framework_instrumentation == {LLMFrameworkEnum.ADK: instrument}
    assert callable(instrument)


async def test_unregistered_framework_loads_its_plugin_entry_point(monkeypatch):
    """A workflow built without plugin discovery still gets the framework's instrumentation."""
    monkeypatch.setattr(framework_instrumentation, "_framework_instrumentation", {})
    calls = []
    requested = []

    class _EntryPoint:
        value = "nat.plugins.strands.register"

        def load(self):
            framework_instrumentation.register_framework_instrumentation(LLMFrameworkEnum.STRANDS)(calls.append)

    def entry_points(**selection):
        requested.append(selection)
        return [_EntryPoint()]

    monkeypatch.setattr(framework_instrumentation.importlib.metadata, "entry_points", entry_points)

    @asynccontextmanager
    async def build_fn(config, builder) -> AsyncIterator[str]:
        yield "workflow"

    wrapped = framework_instrumentation.set_framework_profiler_handler(None, [LLMFrameworkEnum.STRANDS])(build_fn)
    async with wrapped(None, None):
        pass

    assert requested == [{"group": "nat.components", "name": "nat_strands"}]
    assert calls == [None]


async def test_registered_instrumentation_runs_for_detected_framework(monkeypatch):
    calls = []
    monkeypatch.setitem(framework_instrumentation._framework_instrumentation, LLMFrameworkEnum.STRANDS, calls.append)

    @asynccontextmanager
    async def build_fn(config, builder) -> AsyncIterator[str]:
        yield "workflow"

    workflow_llms = {"llm": object()}
    wrapped = framework_instrumentation.set_framework_profiler_handler(workflow_llms,
                                                                       [LLMFrameworkEnum.STRANDS])(build_fn)
    async with wrapped(None, None) as result:
        assert result == "workflow"

    assert calls == [workflow_llms]
