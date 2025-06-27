# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import copy
import typing

from aiq.eval.config import EvaluationRunOutput
from aiq.eval.config import MultiEvaluationRunConfig
from aiq.eval.evaluate import EvaluationRun


class MultiEvaluationRunner:
    """
    Run a multi-evaluation run.
    """

    def __init__(self, config: MultiEvaluationRunConfig):
        """
        Initialize a multi-evaluation run.
        """
        self.config = config
        self.evaluation_run_outputs: dict[typing.Any, EvaluationRunOutput] = {}

    async def run_all(self):
        """
        Run all evaluations defined by the overrides.
        """
        for override_id, override_value in self.config.overrides.items():
            output = await self.run_single_evaluation(override_id, override_value)
            self.evaluation_run_outputs[override_id] = output

    async def run_single_evaluation(self, override_id: typing.Any, override_value: str) -> EvaluationRunOutput:
        """
        Run a single evaluation and return the output.
        """
        config_copy = copy.deepcopy(self.config.base_config)
        config_copy.override = override_value
        config_copy.write_output = self.config.write_output

        evaluation_run = EvaluationRun(config_copy)
        return await evaluation_run.run_and_evaluate()
