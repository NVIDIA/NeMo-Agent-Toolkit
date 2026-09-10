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

import itertools
import time
import typing
import uuid
from unittest.mock import patch

import pytest

from nat.builder.context import Context
from nat.builder.context import ContextState
from nat.builder.function import Function
from nat.builder.intermediate_step_manager import IntermediateStepManager
from nat.data_models.intermediate_step import IntermediateStepPayload
from nat.data_models.intermediate_step import IntermediateStepType
from nat.data_models.span import Span
from nat.data_models.span import SpanContext
from nat.data_models.span import SpanEvent
from nat.observability.exporter_manager import ExporterManager
from nat.runtime.runner import Runner
from nat.utils import providers

# A fixed, valid UUID4-form string used to make every generated identifier predictable.
_FIXED_ID = "12345678-1234-4321-8765-123456789abc"
_FIXED_TIME = 1700000000.5


@pytest.fixture(name="restore_providers", autouse=True)
def restore_providers_fixture():
    """Restore the previously installed providers after each test to avoid cross-test leakage."""
    previous_id = providers.get_id_provider()
    previous_time = providers.get_time_provider()
    try:
        yield
    finally:
        providers.set_id_provider(previous_id)
        providers.set_time_provider(previous_time)


def _sequential_uuid_provider(start: int = 1) -> typing.Callable[[], str]:
    """Return an id provider yielding UUIDs with sequential high 64 bits (valid trace and span ids)."""
    counter = itertools.count(start)
    return lambda: str(uuid.UUID(int=next(counter) << 64))


def test_zero_derived_trace_and_span_ids_are_rejected():
    # The nil UUID derives an all-zero trace ID and an all-zero span ID.
    providers.set_id_provider(lambda: "00000000-0000-0000-0000-000000000000")
    with pytest.raises(ValueError, match="all-zero trace ID"):
        providers.generate_trace_id()
    with pytest.raises(ValueError, match="all-zero span ID"):
        providers.generate_span_id()


def test_high_word_zero_uuid_is_rejected_for_span_ids_only():
    # High 64 bits zero: a valid non-zero trace ID, but an all-zero span ID.
    providers.set_id_provider(lambda: "00000000-0000-0000-0000-000000000001")
    assert providers.generate_trace_id() == 1
    with pytest.raises(ValueError, match="all-zero span ID"):
        providers.generate_span_id()


def test_default_providers_match_uuid4_and_wall_clock():
    value = providers.generate_id()
    assert uuid.UUID(value).version == 4
    assert providers.generate_id() != value

    before = time.time()
    current = providers.current_time()
    after = time.time()
    assert before <= current <= after

    assert providers.get_id_provider() is providers.default_id_provider
    assert providers.get_time_provider() is providers.default_time_provider

    assert 0 < providers.generate_trace_id() < 2**128
    assert 0 < providers.generate_span_id() < 2**64
    assert abs(providers.current_time_ns() / 1e9 - time.time()) < 60.0


def test_setters_install_provider_and_return_previous():

    def _fixed_id() -> str:
        return _FIXED_ID

    def _fixed_time() -> float:
        return _FIXED_TIME

    previous_id = providers.set_id_provider(_fixed_id)
    previous_time = providers.set_time_provider(_fixed_time)

    assert previous_id is providers.default_id_provider
    assert previous_time is providers.default_time_provider
    assert providers.get_id_provider() is _fixed_id
    assert providers.get_time_provider() is _fixed_time

    assert providers.generate_id() == _FIXED_ID
    assert providers.generate_trace_id() == uuid.UUID(_FIXED_ID).int
    assert providers.generate_span_id() == uuid.UUID(_FIXED_ID).int >> 64
    assert providers.current_time() == _FIXED_TIME
    assert providers.current_time_ns() == int(_FIXED_TIME * 1e9)

    assert providers.set_id_provider(previous_id) is _fixed_id
    assert providers.set_time_provider(previous_time) is _fixed_time
    assert providers.generate_id() != _FIXED_ID


def test_integer_id_derivation_requires_uuid_strings():
    providers.set_id_provider(lambda: "not-a-uuid")

    assert providers.generate_id() == "not-a-uuid"

    with pytest.raises(ValueError):
        providers.generate_trace_id()

    with pytest.raises(ValueError):
        providers.generate_span_id()


def test_intermediate_step_payload_defaults_use_installed_providers():
    """Model default factories must resolve the installed providers lazily, not capture them at import."""
    providers.set_id_provider(_sequential_uuid_provider())
    providers.set_time_provider(lambda: _FIXED_TIME)

    first = IntermediateStepPayload(event_type=IntermediateStepType.CUSTOM_START)
    second = IntermediateStepPayload(event_type=IntermediateStepType.CUSTOM_END)

    assert first.UUID == str(uuid.UUID(int=1 << 64))
    assert second.UUID == str(uuid.UUID(int=2 << 64))
    assert first.event_timestamp == _FIXED_TIME
    assert second.event_timestamp == _FIXED_TIME


def test_span_models_use_installed_providers():
    providers.set_id_provider(_sequential_uuid_provider())
    providers.set_time_provider(lambda: _FIXED_TIME)

    span_context = SpanContext()
    assert span_context.trace_id == 1 << 64
    assert span_context.span_id == 2

    event = SpanEvent(name="event")
    assert event.timestamp == int(_FIXED_TIME * 1e9)

    span = Span(name="span")
    assert span.start_time == int(_FIXED_TIME * 1e9)

    span.end()
    assert span.end_time == int(_FIXED_TIME * 1e9)


async def test_push_active_function_uses_installed_id_provider():
    providers.set_id_provider(lambda: _FIXED_ID)

    context = Context.get()
    with context.push_active_function("my_function", input_data=None):
        assert context.active_function.function_id == _FIXED_ID


class _DummyConfig:
    """Mock config for _DummyFunction."""
    name = None
    type = "dummy_workflow"


class _DummyFunction:
    has_single_output = True
    has_streaming_output = True
    instance_name = "workflow"
    display_name = "workflow"
    config = _DummyConfig()

    def convert(self, v, to_type):
        return v

    async def ainvoke(self, _message, to_type=None):
        return {"ok": True}

    async def astream(self, _message, to_type=None):
        yield "chunk-1"


class _DummyExporterManager:

    def start(self, context_state=None):

        class _Ctx:

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                return False

        return _Ctx()


@pytest.mark.parametrize("method", ["result", "result_stream"])
async def test_runner_run_is_deterministic_with_installed_providers(method: str):
    """A minimal run stamps run, trace, and step identifiers plus timestamps from the installed providers."""
    providers.set_id_provider(lambda: _FIXED_ID)
    providers.set_time_provider(lambda: _FIXED_TIME)

    captured: list[IntermediateStepPayload] = []
    original_push = IntermediateStepManager.push_intermediate_step

    def capture_push(self, payload):
        captured.append(payload)
        return original_push(self, payload)

    ctx_state = ContextState.get()
    tkn_run = ctx_state.workflow_run_id.set(None)
    tkn_trace = ctx_state.workflow_trace_id.set(None)

    try:
        with patch.object(IntermediateStepManager, "push_intermediate_step", capture_push):
            runner = Runner(
                "msg",
                typing.cast(Function, _DummyFunction()),
                ctx_state,
                typing.cast(ExporterManager, _DummyExporterManager()),
            )
            async with runner:
                if method == "result":
                    assert await runner.result() == {"ok": True}
                else:
                    assert [c async for c in runner.result_stream()] == ["chunk-1"]
    finally:
        ctx_state.workflow_run_id.reset(tkn_run)
        ctx_state.workflow_trace_id.reset(tkn_trace)

    start = next(p for p in captured if p.event_type == IntermediateStepType.WORKFLOW_START)
    end = next(p for p in captured if p.event_type == IntermediateStepType.WORKFLOW_END)

    for payload in (start, end):
        assert payload.UUID == _FIXED_ID
        assert payload.event_timestamp == _FIXED_TIME
        assert payload.metadata.provided_metadata["workflow_run_id"] == _FIXED_ID
        assert payload.metadata.provided_metadata["workflow_trace_id"] == f"{uuid.UUID(_FIXED_ID).int:032x}"
