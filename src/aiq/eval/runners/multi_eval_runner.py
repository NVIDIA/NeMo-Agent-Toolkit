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

from aiq.eval.config import EvaluationRunConfig
from aiq.eval.config import EvaluationRunOutput
from aiq.eval.evaluate import EvaluationRun
from aiq.eval.runners.config import MultiEvaluationRunConfig


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
        for id, config in self.config.configs.items():
            output = await self.run_single_evaluation(id, config)
            self.evaluation_run_outputs[id] = output

        return self.evaluation_run_outputs

    async def run_single_evaluation(self, id: typing.Any, config: EvaluationRunConfig) -> EvaluationRunOutput:
        """
        Run a single evaluation and return the output.
        """
        # copy the config in case the caller is using the same config for multiple evaluations
        config_copy = copy.deepcopy(config)
        evaluation_run = EvaluationRun(config_copy)
        return await evaluation_run.run_and_evaluate()
