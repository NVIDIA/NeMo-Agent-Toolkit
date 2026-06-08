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

    Maps each top-level field of the function's input or output schema to the dotted
    sub-paths that reach the string(s) to guard. Each string leaf reached is evaluated
    in its own independent rail call, and a non-blocking rewrite is written back to that
    exact leaf so siblings are left untouched.

    Configured in YAML under a function name within ``workflow_functions``::

        workflow_functions:
          retail_tools__get_all_products:
            description: []        # guard the string field ``description`` directly
            review_texts: []       # guard each string in the list field ``review_texts``
          retail_tools__get_product_info:
            reviews:               # guard ``review`` nested in each item of the list ``reviews``
              - review

    Each entry takes one of two forms:
        ``field: []``: Guard the value of the top-level ``field`` itself. The field must
            be a ``str`` or a ``list[str]``; a ``list[str]`` fans out and each element is
            guarded in its own rail call.
        ``field: [sub.path, ...]``: Descend into ``field`` and guard each ``str`` reached
            by every listed dotted ``sub.path``. Any segment that crosses a list field
            fans out, guarding the leaf on each element.

    For example, given a ``get_product_info`` output shaped like::

        {
          "name": "Wireless Mouse",
          "description": "Ergonomic 2.4GHz mouse.",
          "reviews": [
            {"author": "Ada", "rating": 5, "review": "Loved it, works great!"},
            {"author": "Lin", "rating": 2, "review": "Stopped working after a week."},
          ],
        }

    the selection ``reviews: [review]`` reaches the ``review`` string on each item and
    guards each in its own rail call, leaving ``name``, ``description``, and every sibling
    field (``author``, ``rating``) untouched::

        rail call 1: "Loved it, works great!"
        rail call 2: "Stopped working after a week."
    """

    root: dict[str, list[str]] = Field(default_factory=dict)


class GuardrailsMiddlewareConfig(DynamicMiddlewareConfig, name="guardrails"):
    """Guardrails policy attached to a NAT workflow via dynamic middleware."""

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
        description="Lists the workflow functions to wrap and, optionally, which fields of the boundary "
        "input or output value to send to the guardrail.")
    stream_output_rails: bool = Field(
        default=False,
        description="When True, output rails are applied token-by-token as the stream is produced "
        "using ``LLMRails.stream_async()``. The Colang policy must set "
        "``rails.output.streaming.enabled: true``. Incompatible with mapping-form "
        "``workflow_functions`` (field-path selection); use only when ``workflow_functions`` "
        "is a list of function names or omitted entirely.",
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

        if self.stream_output_rails and isinstance(self.workflow_functions, dict):
            raise ValueError("stream_output_rails cannot be used with mapping-form workflow_functions. "
                             "stream_async() evaluates raw token text and has no concept of structured field paths. "
                             "Use a list of function names for workflow_functions, or disable stream_output_rails.")
        return self


__all__ = ["GuardrailFunctionFields", "GuardrailsMiddlewareConfig"]
