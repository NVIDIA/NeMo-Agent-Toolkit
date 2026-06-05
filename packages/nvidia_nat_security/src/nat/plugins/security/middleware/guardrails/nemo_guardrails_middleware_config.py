# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Configuration for NeMo Guardrails middleware."""

from __future__ import annotations

from nemoguardrails import RailsConfig
from pydantic import Field
from pydantic import RootModel
from pydantic import model_validator

from nat.middleware.dynamic.dynamic_middleware_config import DynamicMiddlewareConfig


class GuardrailFunctionFields(RootModel[dict[str, list[str]]]):
    """Field selection for one intercepted function.

    Maps each top-level input or output schema field to the dotted subpaths reaching the
    string(s) to guard. An empty list guards the field itself; a subpath crossing a list
    field fans out and is concatenated into a single rail call.
    """

    root: dict[str, list[str]] = Field(default_factory=dict)


class GuardrailsMiddlewareConfig(DynamicMiddlewareConfig, name="guardrails"):
    """Guardrails policy attached to a NAT workflow via dynamic middleware. # TODO EE: Fix formatted docstrings


    """

    guardrails: RailsConfig | None = Field(
        default=None,
        description="NeMo Guardrails ``RailsConfig`` policy. Mutually exclusive with guardrails_root.",
    )
    guardrails_root: str | None = Field(
        default=None,
        description="Policy directory loaded via ``RailsConfig.from_path``; mutually exclusive with guardrails.",
    )
    llm_bindings: dict[str, str] | None = Field(
        default=None,
        description="Maps Guardrails model type to NAT llms key for model-backed rails.",
    )
    workflow_functions: list[str] | dict[str, GuardrailFunctionFields] | None = Field(
        default=None,
        description="Lists the workflow functions to wrap and, optionally, which of their arguments to "
        "send to the guardrail.\n"
        "As a list of function names, the whole argument of each function is sent to the guardrail.\n"
        "Each entry names an argument and how to reach the string(s) to guard: an empty list selects "
        "the argument itself (e.g. ``description: []``), and a list of dotted paths reaches strings "
        "nested inside it (e.g. ``reviews: [review]`` selects the ``review`` of each item in "
        "``reviews``).",
    )

    @model_validator(mode="after")
    def _finalize_guardrails(self) -> GuardrailsMiddlewareConfig:
        """Load guardrails from guardrails_root when needed and enforce Colang 1.0 at config load."""
        if (self.guardrails is None) == (self.guardrails_root is None):
            raise ValueError("Specify exactly one of guardrails or guardrails_root.")

        if self.guardrails is None:
            from nemoguardrails import RailsConfig as _RailsConfig

            self.guardrails = _RailsConfig.from_path(self.guardrails_root)  # type: ignore[arg-type]

        if self.guardrails.colang_version != "1.0":
            raise ValueError(f"Colang {self.guardrails.colang_version} is not supported; use Colang 1.0.")
        if self.guardrails_root is not None:
            self.guardrails_root = None
        return self


__all__ = ["GuardrailFunctionFields", "GuardrailsMiddlewareConfig"]
