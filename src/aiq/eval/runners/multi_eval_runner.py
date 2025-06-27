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

import typing

from aiq.eval.config import EvaluationRunOutput
from aiq.eval.config import MultiEvalutionRunConfig
from aiq.eval.evaluate import EvaluationRun


class MultiEvaluationRunner:
    """
    Run a multi-evaluation run.
    """

    def __init__(self, config: MultiEvalutionRunConfig):
        """
        Initialize a multi-evaluation run.
        """
        self.config = config
        self.evaluation_run_outputs: dict[typing.Any, EvaluationRunOutput] = {}

    async def run_all(self):
        """
        Run the multi-evaluation run.
        """
        for override_id, override_value in self.config.overrides.items():
            self.evaluation_runs[override_id] = await self.run_single_evaluation(override_id, override_value)

    async def run_single_evaluation(self, override_id: typing.Any, override_value: str):
        """
        Run a single evaluation.
        """
        # Update the base config with the override
        self.config.base_config.override = override_value
        self.config.base_config.write_output = self.config.write_output

        # Instantiate the evaluation run
        evaluation_run = EvaluationRun(self.config.base_config)

        # Run the evaluation
        evaluation_run_output = await evaluation_run.run_and_evaluate()
        self.evaluation_run_outputs[override_id] = evaluation_run_output
