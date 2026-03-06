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

from pathlib import Path

from nat.data_models.evaluate_runtime import EvaluationRunConfig
from nat.eval.eval_callbacks import EvalCallbackManager
from nat.plugins.eval.runtime.evaluate import EvaluationRun


class TestEvaluationRunCallbacks:

    def test_callback_manager_accepted_by_init(self):
        """EvaluationRun accepts callback_manager without error."""
        mgr = EvalCallbackManager()
        config = EvaluationRunConfig(config_file=Path("dummy.yml"))
        runner = EvaluationRun(config=config, callback_manager=mgr)
        assert runner.callback_manager is mgr

    def test_callback_manager_defaults_to_empty(self):
        """EvaluationRun defaults callback_manager to an empty EvalCallbackManager."""
        config = EvaluationRunConfig(config_file=Path("dummy.yml"), write_output=False)
        runner = EvaluationRun(config=config)
        assert isinstance(runner.callback_manager, EvalCallbackManager)
        assert not runner.callback_manager.has_callbacks
