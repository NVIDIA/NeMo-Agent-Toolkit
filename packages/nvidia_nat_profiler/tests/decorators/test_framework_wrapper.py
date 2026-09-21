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

import sys
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from types import ModuleType

from nat.builder.framework_enum import LLMFrameworkEnum
from nat.plugins.profiler.decorators import framework_wrapper


async def test_crewai_profiler_loads_handler_from_real_module(monkeypatch):
    """The CrewAI branch must load the profiler handler from crewai_callback_handler, not callback_handler."""
    instrument_calls = []

    class _StubHandler:

        def instrument(self):
            instrument_calls.append(True)

    stub_module = ModuleType("nat.plugins.crewai.crewai_callback_handler")
    stub_module.CrewAIProfilerHandler = _StubHandler
    monkeypatch.setitem(sys.modules, "nat.plugins.crewai.crewai_callback_handler", stub_module)
    monkeypatch.setitem(framework_wrapper._library_instrumented, "crewai", False)

    @asynccontextmanager
    async def build_fn(config, builder) -> AsyncIterator[str]:
        yield "workflow"

    wrapped = framework_wrapper.set_framework_profiler_handler(None, [LLMFrameworkEnum.CREWAI])(build_fn)
    async with wrapped(None, None) as result:
        assert result == "workflow"

    assert instrument_calls == [True]
    assert framework_wrapper._library_instrumented["crewai"] is True
