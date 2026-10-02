# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Profile rail-owned LangChain models without duplicating workflow callbacks."""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import asynccontextmanager
from typing import Any

from langchain_core.callbacks.base import BaseCallbackManager
from langchain_core.language_models import BaseLanguageModel
from langchain_core.runnables.base import RunnableBindingBase
from langchain_core.runnables.configurable import DynamicRunnable

from nat.builder.context import Context
from nat.data_models.profiler_callback import BaseProfilerCallback
from nat.plugins.langchain.callback_handler import LangchainProfilerHandler


class GuardrailsProfilerHandler(LangchainProfilerHandler):
    """Fallback profiler for models created internally by NeMo Guardrails.

    The workflow configure hook takes precedence at each invocation, including
    when a previously bound model is reused by a different workflow context.
    """

    def __init__(self) -> None:
        super().__init__()
        self._model_profilers: list[LangchainProfilerHandler] = []


@asynccontextmanager
async def profile_rail_calls(handler: GuardrailsProfilerHandler):
    """Close owned model spans even if cancellation interrupts callback setup."""
    parent_span = Context.get().active_span_id
    workflow_handler = handler._workflow_profiler()
    try:
        yield
    except BaseException as error:
        profilers = [handler, *handler._model_profilers]
        if workflow_handler is not None and all(workflow_handler is not profiler for profiler in profilers):
            profilers.append(workflow_handler)
        for profiler in profilers:
            await profiler.close_llm_runs(parent_span, error)
        raise


def attach_rail_profiler(rails: Any, handler: GuardrailsProfilerHandler) -> None:
    """Preserve model callbacks and instrument native and registered rail LLMs."""
    models = [getattr(rails, "llm", None)]
    params = getattr(getattr(rails, "runtime", None), "registered_action_params", {})
    if isinstance(params, Mapping):
        for name, value in params.items():
            if name == "llm" or name.endswith("_llm"):
                models.append(value)
            elif name == "llms" and isinstance(value, Mapping):
                models.extend(value.values())

    seen: set[int] = set()
    while models:
        model = models.pop()
        if model is None or id(model) in seen:
            continue
        seen.add(id(model))
        if not isinstance(model, BaseLanguageModel):
            # Configurable models and runnable bindings retain the actual model.
            if isinstance(model, DynamicRunnable):
                models.append(model.default)
            elif isinstance(model, RunnableBindingBase):
                models.append(model.bound)
            continue
        callbacks = model.callbacks
        existing = callbacks.handlers if isinstance(callbacks, BaseCallbackManager) else callbacks or []
        for callback in existing:
            if isinstance(callback, LangchainProfilerHandler) and callback is not handler:
                if all(callback is not profiler for profiler in handler._model_profilers):
                    handler._model_profilers.append(callback)
        if any(isinstance(callback, BaseProfilerCallback) for callback in existing):
            continue
        if isinstance(callbacks, BaseCallbackManager):
            callbacks = callbacks.copy()
            callbacks.add_handler(handler)
        else:
            callbacks = [*existing, handler]
        model.callbacks = callbacks
