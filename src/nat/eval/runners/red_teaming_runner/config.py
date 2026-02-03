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
from nat.eval.red_teaming_evaluator.register import RedTeamingEvaluatorConfig
from nat.middleware.red_teaming.red_teaming_middleware_config import RedTeamingMiddlewareConfig

logger = logging.getLogger(__name__)


class _RedTeamingScenarioRaw(BaseModel):
    """Private: Scenario with dict evaluator for parsing _extends.

    This type is only used during YAML/JSON parsing when evaluators
    contain _extends references. After validation, all scenarios are
    converted to RedTeamingScenario with proper evaluator configs.
    """

    scenario_id: str | None = Field(default=None, description="Optional unique identifier for this scenario.")

    middleware: RedTeamingMiddlewareConfig | None = Field(default=None,
                                                          description="Full middleware configuration to apply.")

    evaluator: dict[str, typing.Any] = Field(description="Evaluator as dict, potentially with _extends field.")

    tags: list[str] = Field(default=[], description="Tags for bookkeeping and categorization of scenarios.")

    scenario_group: str = Field(
        default="default_scenario_group",
        description="Group name for grouping similar scenarios together for plotting and analysis.")


class RedTeamingScenario(BaseModel):
    """A single red teaming scenario configuration.

    Each scenario defines a complete middleware and evaluator configuration.
    The evaluator can use _extends to inherit from evaluator_defaults.

    Attributes:
        scenario_id: Optional unique identifier. If not provided, the dict key
            from ``RedTeamingRunnerConfig.scenarios`` is used.
        middleware: Full middleware configuration to apply. Set to None for
            baseline scenarios (no middleware modification).
        evaluator: Complete evaluator configuration. Can inherit from
            ``evaluator_defaults`` using ``_extends`` in YAML/JSON.
    """

    scenario_id: str | None = Field(default=None,
                                    description="Optional unique identifier for this scenario. "
                                    "If not provided, the dict key is used.")

    middleware: RedTeamingMiddlewareConfig | None = Field(
        default=None,
        description="Full middleware configuration to apply. "
        "Set to None for baseline scenarios (no middleware modification).")

    evaluator: RedTeamingEvaluatorConfig = Field(description="Complete evaluator configuration for this scenario.")

    tags: list[str] = Field(default=[], description="Tags for bookkeeping and categorization of scenarios.")

    scenario_group: str = Field(
        default="default_scenario_group",
        description="Group name for grouping similar scenarios together for plotting and analysis.")


class RedTeamingRunnerConfig(BaseModel):
    """Top-level configuration for red teaming evaluation.

    Supports ``evaluator_defaults`` for DRY configuration with ``_extends`` inheritance.

    Attributes:
        base_workflow: Optional path to the base workflow configuration file.
        llms: Dictionary of LLM configurations keyed by name.
        evaluator_defaults: Named evaluator configs that scenarios can extend.
        general: General evaluation settings (concurrency, output, dataset).
        scenarios: Dictionary of scenario configurations.

    Example YAML configuration::

        base_workflow: ./configs/base_workflow.yml

        llms:
          judge_llm:
            _type: nim
            model_name: meta/llama-3.1-70b-instruct

        evaluator_defaults:
          standard_eval:
            llm_name: judge_llm
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
            evaluator:
              _extends: standard_eval
              scenario_specific_instructions: "Check for 42.0..."

          custom_scenario:
            tags: [category_1, category_2]
            middleware: {}
            evaluator:
              llm_name: judge_llm
              _type: red_teaming_evaluator
              judge_llm_prompt: "Custom prompt..."
              filter_conditions: []
    """

    base_workflow: Path | None = Field(default=None,
                                       description="Optional path to the base workflow configuration file. "
                                       "Can be overridden by CLI --config_file argument.")

    llms: dict[str, LLMBaseConfig] = Field(description="Dictionary of LLM configurations keyed by name. "
                                           "Scenarios reference these LLMs in their evaluator configs.")

    evaluator_defaults: dict[str, RedTeamingEvaluatorConfig] | None = Field(
        default=None,
        description="Named evaluator defaults that scenarios can extend. "
        "Each must be a complete, valid RedTeamingEvaluatorConfig.")

    general: EvalGeneralConfig | None = Field(default=None,
                                              description="General evaluation settings (concurrency, output, dataset).")

    scenarios: dict[str, RedTeamingScenario | _RedTeamingScenarioRaw] = Field(
        description="Dictionary of scenarios. Pydantic tries RedTeamingScenario first, "
        "falls back to _RedTeamingScenarioRaw for dict-based evaluators with _extends.")

    @model_validator(mode="after")
    def validate_and_resolve_scenarios(self) -> RedTeamingRunnerConfig:
        """Validate scenarios and resolve _extends inheritance.

        This runs after Pydantic parsing, so evaluator_defaults are already
        validated RedTeamingEvaluatorConfig objects. We convert any
        _RedTeamingScenarioRaw to RedTeamingScenario by resolving _extends.

        Returns:
            The validated configuration with all scenarios as RedTeamingScenario
        """
        converted_scenarios: dict[str, RedTeamingScenario] = {}

        for scenario_key, scenario in self.scenarios.items():
            scenario_id = scenario.scenario_id or scenario_key
            scenario.scenario_id = scenario_id

            if isinstance(scenario, _RedTeamingScenarioRaw):
                # Raw scenario with dict evaluator - resolve _extends
                evaluator_dict = scenario.evaluator
                extends_key = evaluator_dict.get("_extends")

                if extends_key:
                    # Validate extends_key exists
                    if not self.evaluator_defaults or extends_key not in self.evaluator_defaults:
                        available = list(self.evaluator_defaults.keys()) if self.evaluator_defaults else []
                        raise ValueError(
                            f"Scenario '{scenario_id}' references evaluator_defaults "
                            f"'{extends_key}' which doesn't exist. Available: {available}."
                            f"If attempting to extend a default evaluator, make sure the required default evaluator is"
                            "defined in the evaluator_defaults section.")

                    # Shallow merge: base config dict + overrides
                    base_config = self.evaluator_defaults[extends_key]
                    base_dict = base_config.model_dump(mode='python')

                    # Remove _extends and apply overrides (shallow merge)
                    overrides = {k: v for k, v in evaluator_dict.items() if k != "_extends"}
                    merged_dict = {**base_dict, **overrides}

                    # Validate merged config
                    evaluator_dict = merged_dict

                scenario_dict = scenario.model_dump(mode='python')
                scenario_dict['evaluator'] = evaluator_dict
                # Create proper RedTeamingScenario
                converted_scenarios[scenario_id] = RedTeamingScenario(**scenario_dict)
            else:
                # Already a proper RedTeamingScenario, ensure scenario_id is set
                if scenario.scenario_id is None:
                    scenario.scenario_id = scenario_id
                converted_scenarios[scenario_id] = scenario

        # Warn if multiple baseline scenarios
        baseline_scenarios = [sid for sid, s in converted_scenarios.items() if s.middleware is None]
        if len(baseline_scenarios) > 1:
            logger.warning(
                "Found %d baseline scenarios (middleware: null): %s. "
                "It's recommended to have only one baseline scenario.",
                len(baseline_scenarios),
                baseline_scenarios)

        # Replace scenarios with fully converted dict
        object.__setattr__(self, 'scenarios', converted_scenarios)
        return self

    @classmethod
    def rebuild_annotations(cls) -> bool:
        """Rebuild field annotations with discriminated unions.

        This method updates the llms dict value annotation to use a
        discriminated union of all registered LLM providers. This allows
        Pydantic to correctly deserialize the _type field into the appropriate
        concrete LLM config class.

        Returns:
            True if the model was rebuilt, False otherwise.
        """
        type_registry = GlobalTypeRegistry.get()

        # Create discriminated union annotation for LLM configs
        LLMAnnotation = typing.Annotated[type_registry.compute_annotation(LLMBaseConfig),
                                         Discriminator(TypedBaseModel.discriminator)]

        should_rebuild = False

        # Update the llms dict annotation
        llms_field = cls.model_fields.get("llms")
        if llms_field is not None:
            expected_annotation = dict[str, LLMAnnotation]
            if llms_field.annotation != expected_annotation:
                llms_field.annotation = expected_annotation
                should_rebuild = True

        if should_rebuild:
            cls.model_rebuild(force=True)
            return True

        return False


# Register hook to rebuild annotations when new types are registered
GlobalTypeRegistry.get().add_registration_changed_hook(lambda: RedTeamingRunnerConfig.rebuild_annotations())

__all__ = [
    "RedTeamingRunnerConfig",
    "RedTeamingScenario",
]
