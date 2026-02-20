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

from unittest.mock import MagicMock

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
