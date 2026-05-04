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

from nat.profiler.parameter_optimization.optimizer_callbacks import OptimizerCallback
from nat.profiler.parameter_optimization.optimizer_callbacks import OptimizerCallbackManager
from nat.profiler.parameter_optimization.optimizer_callbacks import TrialResult


class TestOptimizerCallbackManager:

    def test_on_trial_end(self):
        cb = MagicMock(spec=OptimizerCallback)
        mgr = OptimizerCallbackManager()
        mgr.register(cb)
        result = TrialResult(trial_number=0, parameters={"t": 0.7}, metric_scores={"acc": 0.85}, is_best=True)
        mgr.on_trial_end(result)
        cb.on_trial_end.assert_called_once_with(result)

    def test_on_study_end(self):
        cb = MagicMock(spec=OptimizerCallback)
        mgr = OptimizerCallbackManager()
        mgr.register(cb)
        best = TrialResult(trial_number=0, parameters={"x": 1}, metric_scores={"s": 0.9}, is_best=True)
        mgr.on_study_end(best_trial=best, total_trials=10)
        cb.on_study_end.assert_called_once_with(best_trial=best, total_trials=10)

    def test_callback_error_is_swallowed(self):
        cb = MagicMock(spec=OptimizerCallback)
        cb.on_trial_end.side_effect = RuntimeError("boom")
        mgr = OptimizerCallbackManager()
        mgr.register(cb)
        mgr.on_trial_end(TrialResult(trial_number=0, parameters={}, metric_scores={}, is_best=False))

    def test_empty_manager(self):
        mgr = OptimizerCallbackManager()
        mgr.on_trial_end(TrialResult(trial_number=0, parameters={}, metric_scores={}, is_best=False))

    def test_trial_result_with_prompts(self):
        result = TrialResult(
            trial_number=0,
            parameters={},
            metric_scores={"acc": 0.9},
            is_best=True,
            prompts={"functions.agent.prompt": "You are a helpful assistant."},
        )
        assert result.prompts is not None
        assert "functions.agent.prompt" in result.prompts
