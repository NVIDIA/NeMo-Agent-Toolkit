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

from unittest.mock import AsyncMock
from unittest.mock import MagicMock

import pytest

from nat.eval.eval_callbacks import EvalCallback
from nat.eval.eval_callbacks import EvalCallbackManager
from nat.eval.eval_callbacks import EvalResult
from nat.eval.evaluator.evaluator_model import EvalInputItem


class TestEvalCallbackManager:

    def test_on_eval_complete(self):
        cb = MagicMock(spec=EvalCallback)
        mgr = EvalCallbackManager()
        mgr.register(cb)
        result = EvalResult(metric_scores={"accuracy": 0.85}, items=[])
        mgr.on_eval_complete(result)
        cb.on_eval_complete.assert_called_once_with(result)

    def test_on_dataset_loaded(self):
        cb = MagicMock(spec=EvalCallback)
        mgr = EvalCallbackManager()
        mgr.register(cb)
        items = [EvalInputItem(id="q1", input_obj="2+2", expected_output_obj="4", full_dataset_entry={})]
        mgr.on_dataset_loaded(dataset_name="ds", items=items)
        cb.on_dataset_loaded.assert_called_once_with(dataset_name="ds", items=items)

    def test_multiple_callbacks(self):
        cb1 = MagicMock(spec=EvalCallback)
        cb2 = MagicMock(spec=EvalCallback)
        mgr = EvalCallbackManager()
        mgr.register(cb1)
        mgr.register(cb2)
        result = EvalResult(metric_scores={"s": 0.5}, items=[])
        mgr.on_eval_complete(result)
        cb1.on_eval_complete.assert_called_once()
        cb2.on_eval_complete.assert_called_once()

    def test_callback_error_is_swallowed(self):
        cb = MagicMock(spec=EvalCallback)
        cb.on_eval_complete.side_effect = RuntimeError("boom")
        mgr = EvalCallbackManager()
        mgr.register(cb)
        mgr.on_eval_complete(EvalResult(metric_scores={}, items=[]))  # Should not raise

    def test_empty_manager(self):
        mgr = EvalCallbackManager()
        mgr.on_eval_complete(EvalResult(metric_scores={}, items=[]))  # Should not raise

    def test_optional_sync_hooks(self):
        cb = MagicMock()
        mgr = EvalCallbackManager()
        mgr.register(cb)

        mgr.on_eval_started(workflow_alias="wf", eval_input="ei", config={"a": 1}, job_id="job-1")
        mgr.on_prediction(item={"id": 1}, output="out")
        mgr.on_eval_summary(usage_stats={"runtime": 1.0}, evaluation_results=[], profiler_results={})

        cb.on_eval_started.assert_called_once()
        cb.on_prediction.assert_called_once()
        cb.on_eval_summary.assert_called_once()

    @pytest.mark.asyncio
    async def test_optional_async_hooks(self):
        cb = MagicMock()
        cb.a_on_usage_stats = AsyncMock()
        cb.a_on_evaluator_score = AsyncMock()
        cb.a_on_export_flush = AsyncMock()

        mgr = EvalCallbackManager()
        mgr.register(cb)

        await mgr.a_on_usage_stats(item={"id": 1}, usage_stats_item={"runtime": 0.1})
        await mgr.a_on_evaluator_score(eval_output={"score": 0.9}, evaluator_name="acc")
        await mgr.a_on_export_flush()

        cb.a_on_usage_stats.assert_awaited_once()
        cb.a_on_evaluator_score.assert_awaited_once()
        cb.a_on_export_flush.assert_awaited_once()

    def test_evaluation_context_optional(self):

        class _DummyContext:

            def __init__(self):
                self.entered = False

            def __enter__(self):
                self.entered = True
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

        cb = MagicMock()
        ctx = _DummyContext()
        cb.evaluation_context.return_value = ctx

        mgr = EvalCallbackManager()
        mgr.register(cb)

        with mgr.evaluation_context():
            pass

        assert ctx.entered is True
