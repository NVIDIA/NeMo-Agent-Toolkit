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

from abc import ABC
from abc import abstractmethod
from typing import Any

from nat.data_models.finetuning import FinetuneConfig
from nat.data_models.finetuning import TrajectoryBuilderConfig
from nat.data_models.finetuning import TrajectoryCollection
from nat.eval.config import EvaluationRunOutput
from nat.eval.evaluator.evaluator_model import EvalOutputItem
from nat.utils.io.supress_logs import suppress_logs


class TrajectoryBuilder(ABC):
    """
    Abstract interface for building trajectories from episode items.
    """

    def __init__(self, trajectory_builder_config: TrajectoryBuilderConfig):
        self.trajectory_builder_config = trajectory_builder_config
        self.run_config: FinetuneConfig = None

    async def initialize(self, run_config: FinetuneConfig) -> None:
        """
        Asynchronously initialize any resources needed for the trajectory builder.
        """
        self.run_config = run_config
        self.trajectory_builder_config.reward = self.run_config.reward_function

    async def run_eval(self) -> EvaluationRunOutput:
        """
        Run NAT Evaluation to generate episode items for trajectory building.

        Returns:
            EvaluationRunOutput: The output of the evaluation run.
        """

        from nat.eval.evaluate import EvaluationRun
        from nat.eval.evaluate import EvaluationRunConfig

        eval_cfg = EvaluationRunConfig(config_file=self.run_config.run_configuration.config_file,
                                       dataset=self.run_config.run_configuration.dataset,
                                       result_json_path=self.run_config.run_configuration.result_json_path,
                                       endpoint=self.run_config.run_configuration.endpoint,
                                       endpoint_timeout=self.run_config.run_configuration.endpoint_timeout,
                                       override=self.run_config.run_configuration.override)

        async with suppress_logs(prefix="nat.eval"):
            evaluation_output = await EvaluationRun(config=eval_cfg).run_and_evaluate()

        return evaluation_output

    @abstractmethod
    async def start_run(self, run_id: str, meta: dict | None = None) -> None:
        """
        Initialize any resources needed for the trajectory builder.

        Args:
            run_id (str): The unique identifier for the training run.
            meta (dict): Metadata associated with the training run.
        """
        raise NotImplementedError

    @abstractmethod
    async def finalize(self, run_id: str, meta: dict | None = None) -> TrajectoryCollection:
        """
        Finalize the trajectory building process and return the constructed trajectories.

        Args:
            run_id (str): The unique identifier for the training run.
            meta (dict): Metadata associated with the training run.

        Returns:
            list[Trajectory]: The list of constructed trajectories.
        """
        raise NotImplementedError

    async def compute_reward(self, output_item: EvalOutputItem, meta: dict | None = None):
        """
        Compute reward for a given EvalOutputItem.

        Args:
            output_item (EvalOutputItem): The evaluation output item.
            meta (dict): Metadata associated with the training run.

        Returns:
            float: The computed reward.
        """
        return float(output_item.score) if output_item.score is not None else 0.0

    @abstractmethod
    def log_progress(self, run_id: str, metrics: dict[str, Any], output_dir: str | None = None) -> None:
        """
        Log trajectory building progress.

        Args:
            run_id: The training run ID
            metrics: Dictionary of metrics to log
            output_dir: Optional output directory override
        """
        raise NotImplementedError
