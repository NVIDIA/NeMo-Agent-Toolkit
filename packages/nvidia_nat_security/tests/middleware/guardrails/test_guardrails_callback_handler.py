# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exercise rail model callbacks through the NeMo Guardrails SDK call path."""

import asyncio
from types import SimpleNamespace

import pytest
from langchain_core.callbacks import AsyncCallbackHandler
from langchain_core.callbacks import CallbackManager
from langchain_core.language_models import BaseChatModel
from langchain_core.language_models.fake import FakeListLLM
from langchain_core.messages import AIMessage
from langchain_core.messages import AIMessageChunk
from langchain_core.outputs import ChatGeneration
from langchain_core.outputs import ChatGenerationChunk
from langchain_core.outputs import ChatResult
from nemoguardrails.actions.llm.utils import llm_call
from nemoguardrails.streaming import StreamingHandler

from nat.builder.context import Context
from nat.data_models.intermediate_step import IntermediateStepType
from nat.middleware.telemetry import trace_middleware
from nat.plugins.langchain.callback_handler import LangchainProfilerHandler
from nat.plugins.security.middleware.guardrails.callback_handler import GuardrailsProfilerHandler
from nat.plugins.security.middleware.guardrails.callback_handler import attach_rail_profiler
from nat.plugins.security.middleware.guardrails.callback_handler import profile_rail_calls


class RailModel(BaseChatModel):
    failure: bool = False
    wait: bool = False

    @property
    def _llm_type(self):
        return "rail-test-model"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        if self.failure:
            raise ValueError("rail model failed")
        return ChatResult(generations=[
            ChatGeneration(message=AIMessage(content="safe",
                                             usage_metadata={
                                                 "input_tokens": 2, "output_tokens": 1, "total_tokens": 3
                                             }))
        ])

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
        if self.wait:
            await asyncio.Event().wait()
        return self._generate(messages, stop, run_manager, **kwargs)

    def _stream(self, messages, stop=None, run_manager=None, **kwargs):
        yield ChatGenerationChunk(message=AIMessageChunk(content="safe"))
        yield ChatGenerationChunk(message=AIMessageChunk(content="",
                                                         usage_metadata={
                                                             "input_tokens": 2, "output_tokens": 1, "total_tokens": 3
                                                         }))


@pytest.fixture
def steps():
    context = Context.get()
    initial_span = context.active_span_id
    recorded = []
    subscription = context.intermediate_step_manager.subscribe(recorded.append)
    yield recorded
    subscription.unsubscribe()
    assert context.active_span_id == initial_span


def rails_with(model):
    return SimpleNamespace(llm=model,
                           runtime=SimpleNamespace(registered_action_params={
                               "llm": model, "safety_llm": model, "llms": {
                                   "main": model, "safety": model
                               }
                           }))


def llm_steps(steps):
    return [
        step for step in steps
        if step.payload.event_type in {IntermediateStepType.LLM_START, IntermediateStepType.LLM_END}
    ]


@pytest.mark.parametrize("plain", [False, True])
async def test_native_and_registered_models_emit_nested_llm_steps(plain, steps):
    model = FakeListLLM(responses=["safe"]) if plain else RailModel()
    handler = GuardrailsProfilerHandler()
    rails = rails_with(model)
    attach_rail_profiler(rails, handler)
    attach_rail_profiler(rails, handler)
    assert model.callbacks == [handler]
    with trace_middleware("guardrails.input", input_data="prompt", function_name="echo", guardrail=True):
        assert await llm_call(model, "prompt") == "safe"
    start, end = llm_steps(steps)
    assert start.UUID == end.UUID
    assert start.parent_id == steps[0].UUID
    assert end.payload.data.output == "safe"
    assert start.payload.name
    if not plain:
        assert end.payload.usage_info.token_usage.total_tokens == 3
    assert not handler._run_id_to_model_name


async def test_live_sdk_stream_emits_tokens_and_usage(steps):
    model = RailModel()
    handler = GuardrailsProfilerHandler()
    attach_rail_profiler(rails_with(model), handler)
    with trace_middleware("guardrails.output.stream", input_data="prompt", function_name="echo", guardrail=True):
        assert await llm_call(model, "prompt", streaming_handler=StreamingHandler()) == "safe"
    start, end = llm_steps(steps)
    assert start.parent_id == steps[0].UUID
    assert end.payload.data.output == "safe"
    assert end.payload.usage_info.token_usage.total_tokens == 3
    assert any(step.payload.event_type == IntermediateStepType.LLM_NEW_TOKEN for step in steps)
    assert not handler._run_id_to_model_name


@pytest.mark.parametrize("manager", [False, True])
def test_existing_callbacks_are_preserved_without_mutating_manager(manager):
    callback = AsyncCallbackHandler()
    original = CallbackManager([callback]) if manager else [callback]
    model = RailModel(callbacks=original)
    handler = GuardrailsProfilerHandler()
    attach_rail_profiler(rails_with(model), handler)
    actual = model.callbacks.handlers if manager else model.callbacks
    assert actual == [callback, handler]
    assert (original.handlers if manager else original) == [callback]


def test_existing_profiler_and_configurable_model_are_not_duplicated():
    handler = LangchainProfilerHandler()
    model = RailModel(callbacks=[handler])
    rails = rails_with(model.bind(temperature=0))
    attach_rail_profiler(rails, GuardrailsProfilerHandler())
    assert model.callbacks == [handler]


@pytest.mark.parametrize("cancel", [False, True])
@pytest.mark.parametrize("existing_profiler", [False, True])
async def test_failure_and_cancellation_close_llm_and_guardrail_spans(cancel, existing_profiler, steps):
    existing = LangchainProfilerHandler() if existing_profiler else None
    model = RailModel(failure=not cancel, wait=cancel, callbacks=[existing] if existing else None)
    handler = GuardrailsProfilerHandler()
    attach_rail_profiler(rails_with(model), handler)

    async def run():
        with trace_middleware("guardrails.input", input_data="prompt", function_name="echo", guardrail=True):
            async with profile_rail_calls(handler):
                await llm_call(model, "prompt")

    if cancel:
        task = asyncio.create_task(run())
        while not llm_steps(steps):
            await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    else:
        with pytest.raises(Exception, match="rail model failed"):
            await run()
    start, end = llm_steps(steps)
    assert start.UUID == end.UUID
    assert end.payload.metadata["status"] == ("CancelledError" if cancel else "ValueError")
    assert not handler._run_id_to_model_name
    if existing is not None:
        assert not existing._run_id_to_model_name
    assert steps[-1].payload.event_type == IntermediateStepType.GUARDRAIL_END


async def test_active_workflow_hook_takes_precedence_on_reused_model(monkeypatch, steps):
    framework = pytest.importorskip("nat.plugins.profiler.decorators.framework_wrapper")
    from langchain_core.tracers.context import register_configure_hook

    register_configure_hook(framework.callback_handler_var, inheritable=True)
    monkeypatch.setitem(framework._library_instrumented, "langchain", True)
    fallback = GuardrailsProfilerHandler()
    model = RailModel()
    attach_rail_profiler(rails_with(model), fallback)
    for _ in range(2):
        profiler = LangchainProfilerHandler()
        token = framework.callback_handler_var.set(profiler)
        try:
            with trace_middleware("guardrails.input", input_data="prompt", function_name="echo", guardrail=True):
                assert await llm_call(model, "prompt") == "safe"
        finally:
            framework.callback_handler_var.reset(token)
        assert not fallback._run_id_to_model_name
        assert not profiler._run_id_to_model_name
    events = llm_steps(steps)
    assert len(events) == 4
    assert events[0].UUID == events[1].UUID
    assert events[2].UUID == events[3].UUID


@pytest.mark.parametrize("plain", [False, True])
async def test_existing_model_profiler_yields_to_active_workflow_hook(plain, monkeypatch, steps):
    """A persistent local profiler and a new workflow hook produce one pair per call."""
    framework = pytest.importorskip("nat.plugins.profiler.decorators.framework_wrapper")
    from langchain_core.tracers.context import register_configure_hook

    register_configure_hook(framework.callback_handler_var, inheritable=True)
    monkeypatch.setitem(framework._library_instrumented, "langchain", True)
    local = LangchainProfilerHandler()
    model = FakeListLLM(responses=["safe"], callbacks=[local]) if plain else RailModel(callbacks=[local])
    fallback = GuardrailsProfilerHandler()
    attach_rail_profiler(rails_with(model), fallback)
    assert model.callbacks == [local]
    policy_spans = []
    for _ in range(2):
        workflow = LangchainProfilerHandler()
        token = framework.callback_handler_var.set(workflow)
        try:
            with trace_middleware("guardrails.input", input_data="prompt", function_name="echo", guardrail=True):
                policy_spans.append(Context.get().active_span_id)
                async with profile_rail_calls(fallback):
                    assert await llm_call(model, "prompt") == "safe"
        finally:
            framework.callback_handler_var.reset(token)
        for handler in (local, fallback, workflow):
            assert not handler._run_id_to_model_name
            assert not handler._run_id_to_llm_input
            assert not handler._run_id_to_parent_span
            assert not handler._run_id_to_start_time

    events = llm_steps(steps)
    assert len(events) == 4
    for index, policy_span in enumerate(policy_spans):
        start, end = events[2 * index:2 * index + 2]
        assert start.UUID == end.UUID
        assert start.parent_id == end.parent_id == policy_span
        assert start.UUID != start.parent_id
    assert events[0].UUID != events[2].UUID
