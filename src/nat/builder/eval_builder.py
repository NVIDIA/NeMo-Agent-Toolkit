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

import dataclasses
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from nat.builder.builder import EvalBuilder
from nat.builder.evaluator import EvaluatorInfo
from nat.builder.framework_enum import LLMFrameworkEnum
from nat.builder.workflow_builder import WorkflowBuilder
from nat.cli.type_registry import TypeRegistry
from nat.data_models.config import Config
from nat.data_models.config import GeneralConfig
from nat.data_models.evaluate import EvalGeneralConfig
from nat.data_models.evaluator import EvaluatorBaseConfig
from nat.data_models.function import EmptyFunctionConfig
from nat.utils.type_utils import override

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class ConfiguredEvaluator:
    config: EvaluatorBaseConfig
    instance: EvaluatorInfo


class WorkflowEvalBuilder(WorkflowBuilder, EvalBuilder):

    def __init__(self,
                 general_config: GeneralConfig | None = None,
                 eval_general_config: EvalGeneralConfig | None = None,
                 registry: TypeRegistry | None = None):
        super().__init__(general_config=general_config, registry=registry)
        self.eval_general_config = eval_general_config
        self._evaluators: dict[str, ConfiguredEvaluator] = {}

    @override
    async def add_evaluator(self, name: str, config: EvaluatorBaseConfig):
        if name in self._evaluators:
            raise ValueError(f"Evaluator `{name}` already exists in the list of evaluators")

        try:
            evaluator_info = self._registry.get_evaluator(type(config))
            info_obj = await self._get_exit_stack().enter_async_context(evaluator_info.build_fn(config, self))

            # Store the evaluator
            self._evaluators[name] = ConfiguredEvaluator(config=config, instance=info_obj)
        except Exception as e:
            logger.error("Error %s adding evaluator `%s` with config `%s`", e, name, config)
            raise

    @override
    def get_evaluator(self, evaluator_name: str) -> EvaluatorInfo:

        if (evaluator_name not in self._evaluators):
            raise ValueError(f"Evaluator `{evaluator_name}` not found")

        return self._evaluators[evaluator_name].instance

    @override
    def get_evaluator_config(self, evaluator_name: str) -> EvaluatorBaseConfig:

        if evaluator_name not in self._evaluators:
            raise ValueError(f"Evaluator `{evaluator_name}` not found")

        # Return the tool configuration object
        return self._evaluators[evaluator_name].config

    @override
    def get_max_concurrency(self) -> int:
        return self.eval_general_config.max_concurrency

    @override
    def get_output_dir(self) -> Path:
        return self.eval_general_config.output_dir

    @override
    def get_all_tools(self, wrapper_type: LLMFrameworkEnum | str):
        tools = []
        tool_wrapper_reg = self._registry.get_tool_wrapper(llm_framework=wrapper_type)
        for fn_name in self._functions:
            fn = self.get_function(fn_name)
            try:
                tools.append(tool_wrapper_reg.build_fn(fn_name, fn, self))
            except Exception:
                logger.exception("Error fetching tool `%s`", fn_name)

        return tools

    def _log_build_failure_evaluator(self,
                                     failing_evaluator_name: str,
                                     completed_evaluators: list[str],
                                     remaining_evaluators: list[str],
                                     original_error: Exception) -> None:
        """
        Log comprehensive evaluator build failure information.

        Args:
            failing_evaluator_name (str): The name of the evaluator that failed to build
            completed_evaluators (list[str]): List of evaluator names that were successfully built
            remaining_evaluators (list[str]): List of evaluator names still to be built
            original_error (Exception): The original exception that caused the failure
        """
        # Convert evaluator names to (name, type) tuples for consistent logging
        completed_components = [(name, "evaluator") for name in completed_evaluators]
        remaining_components = [(name, "evaluator") for name in remaining_evaluators]

        # Use the inherited common logging method from WorkflowBuilder
        self._log_build_failure(failing_evaluator_name,
                                "evaluator",
                                completed_components,
                                remaining_components,
                                original_error)

    async def populate_builder(self, config: Config):
        # Skip setting workflow if workflow config is EmptyFunctionConfig
        skip_workflow = isinstance(config.workflow, EmptyFunctionConfig)

        await super().populate_builder(config, skip_workflow)

        # Initialize progress tracking for evaluators
        completed_evaluators = []
        remaining_evaluators = list(config.eval.evaluators.keys())

        # Instantiate the evaluators with enhanced error logging
        for name, evaluator_config in config.eval.evaluators.items():
            try:
                # Remove from remaining as we start building
                remaining_evaluators.remove(name)

                await self.add_evaluator(name, evaluator_config)

                # Add to completed after successful build
                completed_evaluators.append(name)

            except Exception as e:
                self._log_build_failure_evaluator(name, completed_evaluators, remaining_evaluators, e)
                raise

    @classmethod
    @asynccontextmanager
    async def from_config(cls, config: Config):

        async with cls(config.general, config.eval.general, registry=None) as builder:
            await builder.populate_builder(config)
            yield builder
