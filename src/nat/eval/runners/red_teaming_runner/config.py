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

"""Red teaming runner configuration models.

This module provides configuration models for red teaming evaluation workflows.
The RedTeamingRunnerConfig encapsulates all settings needed to run red teaming
evaluations across multiple scenarios without requiring modifications to the
base workflow.
"""

from __future__ import annotations

import logging
import typing
from pathlib import Path

from pydantic import BaseModel
from pydantic import Discriminator
from pydantic import Field
from pydantic import model_validator

from nat.cli.type_registry import GlobalTypeRegistry
from nat.data_models.common import TypedBaseModel
from nat.data_models.evaluate import EvalGeneralConfig
from nat.data_models.llm import LLMBaseConfig
from nat.eval.red_teaming_evaluator.filter_conditions import IntermediateStepsFilterCondition
from nat.eval.red_teaming_evaluator.register import RedTeamingEvaluatorConfig
from nat.middleware.red_teaming_middleware_config import RedTeamingMiddlewareConfig

logger = logging.getLogger(__name__)

# Fixed LLM name for red teaming evaluator
RED_TEAMING_EVALUATOR_LLM_NAME = "red_teaming_evaluator_llm"


class RedTeamingScenario(BaseModel):
    """A single red teaming scenario configuration.

    Each scenario defines a middleware configuration to apply and optional
    evaluation overrides. The middleware is attached to all functions,
    function_groups, and workflow - the middleware's internal targeting
    handles which functions are actually affected at runtime.

    Attributes:
        scenario_id: Optional unique identifier. If not provided, the dict key
            from RedTeamingRunnerConfig.scenarios is used.
        middleware: Full middleware configuration to apply. Set to None for
            baseline scenarios (no middleware modification).
        evaluation_instructions: Optional scenario-specific instructions that
            override the evaluator's scenario_specific_instructions.
        filter_conditions: Optional filter conditions that override the
            evaluator's default filter conditions.
    """

    scenario_id: str | None = Field(
        default=None,
        description="Optional unique identifier for this scenario. "
        "If not provided, the dict key is used."
    )

    middleware: RedTeamingMiddlewareConfig | None = Field(
        default=None,
        description="Full middleware configuration to apply. "
        "Set to None for baseline scenarios (no middleware modification)."
    )

    evaluation_instructions: str | None = Field(
        default=None,
        description="Scenario-specific instructions for the evaluator. "
        "Overrides the evaluator's scenario_specific_instructions."
    )

    filter_conditions: list[IntermediateStepsFilterCondition] | None = Field(
        default=None,
        description="Filter conditions for selecting intermediate steps to evaluate. "
        "Overrides the evaluator's default filter conditions."
    )


class RedTeamingRunnerConfig(BaseModel):
    """Top-level configuration for red teaming evaluation.

    This configuration encapsulates all settings needed to run red teaming
    evaluations across multiple scenarios. It can optionally reference a
    base workflow, or the base workflow can be provided via CLI.

    The evaluator LLM is automatically named 'red_teaming_evaluator_llm' in the
    generated configs to avoid conflicts with workflow LLMs.

    Attributes:
        base_workflow: Optional path to the base workflow configuration file.
            Can be overridden by CLI --config_file argument.
        evaluator_llm: LLM configuration for the red teaming evaluator.
            Will be added to configs as 'red_teaming_evaluator_llm'.
        evaluator: Default red teaming evaluator configuration. Scenarios
            can override specific fields like evaluation_instructions and
            filter_conditions.
        general: General evaluation settings (concurrency, output, dataset).
        scenarios: Dictionary of scenarios keyed by scenario name. The key
            becomes the scenario_id if not explicitly set in the scenario.

    Example YAML configuration:

        base_workflow: ./configs/base_workflow.yml

        evaluator_llm:
          _type: nim
          model_name: mistralai/mixtral-8x22b-instruct-v0.1

        evaluator:
          _type: red_teaming_evaluator
          judge_llm_prompt: >
            Evaluate whether the attack was successful...
          filter_conditions:
            - name: workflow_output
              event_type: FUNCTION_END
              payload_name: <workflow>
          reduction_strategy: last

        general:
          max_concurrency: 4
          output_dir: ./.tmp/nat/redteaming/

        scenarios:
          intercept_payload_42:
            middleware:
              _type: red_teaming
              target_function_or_group: my_calculator
              attack_payload: "42.0"
            evaluation_instructions: "Check if calculator returns 42.0..."

          baseline:
            middleware: null  # No middleware - baseline scenario
    """

    base_workflow: Path | None = Field(
        default=None,
        description="Optional path to the base workflow configuration file. "
        "Can be overridden by CLI --config_file argument."
    )

    evaluator_llm: LLMBaseConfig = Field(
        description="LLM configuration for the red teaming evaluator. "
        "Will be added to configs as 'red_teaming_evaluator_llm'."
    )

    evaluator: RedTeamingEvaluatorConfig = Field(
        description="Default red teaming evaluator configuration. "
        "Scenarios can override evaluation_instructions and filter_conditions."
    )

    general: EvalGeneralConfig | None = Field(
        default=None,
        description="General evaluation settings (concurrency, output, dataset)."
    )

    scenarios: dict[str, RedTeamingScenario] = Field(
        description="Dictionary of scenarios keyed by scenario name. "
        "The key becomes the scenario_id if not explicitly set."
    )

    @model_validator(mode="after")
    def validate_scenarios(self) -> RedTeamingRunnerConfig:
        """Validate the red teaming configuration.

        Performs the following validations:
        - Warns if multiple baseline scenarios (middleware: null) exist
        - Ensures scenario_ids are set from dict keys if not provided

        Returns:
            The validated configuration
        """
        # Set scenario_id from dict key if not explicitly provided
        for scenario_key, scenario in self.scenarios.items():
            if scenario.scenario_id is None:
                scenario.scenario_id = scenario_key

        # Warn if multiple baseline scenarios
        baseline_scenarios = [
            scenario_id for scenario_id, scenario in self.scenarios.items()
            if scenario.middleware is None
        ]
        if len(baseline_scenarios) > 1:
            logger.warning(
                "Found %d baseline scenarios (middleware: null): %s. "
                "It's recommended to have only one baseline scenario.",
                len(baseline_scenarios),
                baseline_scenarios
            )

        return self

    @classmethod
    def rebuild_annotations(cls) -> bool:
        """Rebuild field annotations with discriminated unions.

        This method updates the evaluator_llm field annotation to use a
        discriminated union of all registered LLM providers. This allows
        Pydantic to correctly deserialize the _type field into the appropriate
        concrete LLM config class.

        Returns:
            True if the model was rebuilt, False otherwise.
        """
        type_registry = GlobalTypeRegistry.get()

        # Create discriminated union annotation for LLM configs
        LLMAnnotation = typing.Annotated[
            type_registry.compute_annotation(LLMBaseConfig),
            Discriminator(TypedBaseModel.discriminator)
        ]

        should_rebuild = False

        evaluator_llm_field = cls.model_fields.get("evaluator_llm")
        if evaluator_llm_field is not None and evaluator_llm_field.annotation != LLMAnnotation:
            evaluator_llm_field.annotation = LLMAnnotation
            should_rebuild = True

        if should_rebuild:
            cls.model_rebuild(force=True)
            return True

        return False


# Register hook to rebuild annotations when new types are registered
GlobalTypeRegistry.get().add_registration_changed_hook(lambda: RedTeamingRunnerConfig.rebuild_annotations())


__all__ = [
    "RED_TEAMING_EVALUATOR_LLM_NAME",
    "RedTeamingRunnerConfig",
    "RedTeamingScenario",
]

