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

import asyncio
import contextvars
from collections.abc import Callable
from enum import StrEnum
from typing import Any

from langsmith.run_helpers import tracing_context
from typing_extensions import override

from nat.data_models.evaluator import EvalInputItem
from nat.data_models.evaluator import EvalOutputItem
from nat.plugins.eval.evaluator.base_evaluator import BaseEvaluator

from .utils import eval_input_item_to_openevals_kwargs
from .utils import eval_input_item_to_run_and_example
from .utils import langsmith_result_to_eval_output_item


class _EvaluatorConvention(StrEnum):
    """Detected evaluator calling convention."""

    RUN_EVALUATOR_CLASS = "run_evaluator_class"
    RUN_EXAMPLE_FUNCTION = "run_example_function"
    OPENEVALS_FUNCTION = "openevals_function"


async def _invoke_maybe_sync(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    """Invoke *fn* with the given arguments, adapting sync callables to async.

    If *fn* is a coroutine function it is awaited directly.  Otherwise it is
    dispatched to the default executor so that it never blocks the event loop.

    The current :mod:`contextvars` context is explicitly copied into the
    executor thread so that caller-side context managers (e.g.,
    ``tracing_context(enabled=False)``) remain effective.
    """
    if asyncio.iscoroutinefunction(fn):
        return await fn(*args, **kwargs)

    ctx = contextvars.copy_context()
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, lambda: ctx.run(fn, *args, **kwargs))


class LangSmithEvaluatorAdapter(BaseEvaluator):
    """NAT evaluator adapter that wraps a LangSmith/openevals evaluator callable.

    Adapts various LangSmith evaluator calling conventions into NAT's
    ``BaseEvaluator`` interface:

    - RunEvaluator: calls ``aevaluate_run`` with synthetic Run/Example objects
    - ``(run, example)`` functions: constructs synthetic Run/Example objects
    - ``(inputs, outputs, reference_outputs)`` functions: passes kwargs directly

    All evaluator calls are wrapped in ``tracing_context(enabled=False)``
    so that LangSmith auto-tracing does not produce unintended traces.
    NAT's own observability pipeline (OTEL-based LangSmith exporter)
    handles tracing separately.
    """

    def __init__(
        self,
        evaluator: Any,
        convention: str,
        max_concurrency: int = 4,
        evaluator_name: str = "langsmith",
        extra_fields: dict[str, str] | None = None,
        score_field: str | None = None,
    ):
        super().__init__(max_concurrency=max_concurrency, tqdm_desc=f"LangSmith ({evaluator_name})")
        self._evaluator = evaluator
        try:
            self._convention = _EvaluatorConvention(convention)
        except ValueError:
            raise ValueError(f"Unknown evaluator convention '{convention}'. "
                             f"Expected one of: {[e.value for e in _EvaluatorConvention]}") from None
        self._extra_fields = extra_fields
        self._score_field = score_field

    @override
    async def evaluate_item(self, item: EvalInputItem) -> EvalOutputItem:
        """Evaluate a single item using the wrapped evaluator."""
        if self._convention == _EvaluatorConvention.RUN_EVALUATOR_CLASS:
            result = await self._call_run_evaluator(item)
        elif self._convention == _EvaluatorConvention.RUN_EXAMPLE_FUNCTION:
            result = await self._call_run_example_function(item)
        else:
            result = await self._call_openevals_function(item)

        return langsmith_result_to_eval_output_item(
            item.id,
            result,
            score_field=self._score_field,
        )

    async def _call_run_evaluator(self, item: EvalInputItem) -> Any:
        """Call a RunEvaluator subclass instance via ``aevaluate_run``."""
        run, example = eval_input_item_to_run_and_example(item)

        with tracing_context(enabled=False):
            return await self._evaluator.aevaluate_run(run, example)

    async def _call_run_example_function(self, item: EvalInputItem) -> Any:
        """Call a function with ``(run, example)`` signature."""
        run, example = eval_input_item_to_run_and_example(item)

        with tracing_context(enabled=False):
            return await _invoke_maybe_sync(self._evaluator, run, example)

    async def _call_openevals_function(self, item: EvalInputItem) -> Any:
        """Call a function with ``(inputs, outputs, reference_outputs)`` signature."""
        kwargs = eval_input_item_to_openevals_kwargs(item, extra_fields=self._extra_fields)

        with tracing_context(enabled=False):
            return await _invoke_maybe_sync(self._evaluator, **kwargs)
