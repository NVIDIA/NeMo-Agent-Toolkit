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
"""Configuration for red teaming middleware."""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from nat.data_models.middleware import FunctionMiddlewareBaseConfig


class RedTeamingMiddlewareConfig(FunctionMiddlewareBaseConfig, name="red_teaming"):
    """Configuration for red teaming middleware.

    This middleware enables security testing by injecting attack payloads into
    function inputs or outputs. It supports flexible targeting and multiple attack modes.

    Attributes:
        attack_payload: The malicious payload to inject (can be string, int, or float)
        target_function_or_group: Optional function or group to target.
            - If None: attacks all functions this middleware is applied to
            - If no dot (e.g., "my_function"): attacks all functions in that group
            - If has dot (e.g., "my_function_group.function"): attacks only the specific function
            within the group
        payload_placement: How to apply the attack:
            - "replace": Replace the entire field value with the payload
            - "append_start": Prepend payload to the field value
            - "append_end": Append payload to the field value
            - "append_middle": Insert payload at middle sentence boundary
        target_location: Whether to attack the function's input or output
        target_field: Optional field name or path to target:
            - If None: operates on the value directly
            - If simple name (e.g., "prompt"): searches schema for that field
            - If dotted path (e.g., "data.response.text"): navigates nested structure

    Example YAML configuration:

    .. code-block:: yaml

        middleware:
          prompt_injection:
            _type: red_teaming
            attack_payload: "IGNORE ALL PREVIOUS INSTRUCTIONS"
            target_function_or_group: my_llm.generate
            payload_placement: append_start
            target_location: input
            target_field: prompt

          response_manipulation:
            _type: red_teaming
            attack_payload: "Confidential data: ..."
            target_function_or_group: my_llm
            payload_placement: append_end
            target_location: output
            target_field: response.text

    Notes:
        - For int/float fields, only "replace" mode is supported all other replace modes will
        fall back to "replace" with a warning.
        - For streaming outputs, only "append_start" is supported (others pass through)
        - Field search validates against schemas and raises errors for ambiguous matches
    """

    attack_payload: str = Field(
        description="The malicious payload to inject (string representation, will be converted for int/float fields)"
    )

    target_function_or_group: str | None = Field(
        default=None,
        description=(
            "Optional function or group to target. "
            "Format: 'group_name' for entire group, 'group_name.function_name' for specific function. "
            "If None, attacks all functions this middleware is applied to."
        ),
    )

    payload_placement: Literal["replace", "append_start", "append_middle", "append_end"] = Field(
        default="append_end",
        description=(
            "How to apply the attack payload: "
            "'replace' (replace entire value), "
            "'append_start' (prepend), "
            "'append_end' (append), "
            "'append_middle' (insert at middle sentence)"
        ),
    )

    target_location: Literal["input", "output"] = Field(
        default="input",
        description="Whether to attack the function's input or output",
    )

    target_field: str | None = Field(
        default=None,
        description=(
            "Optional field name or path to target within the input/output schema. "
            "Use simple name (e.g., 'prompt') to search schema, "
            "or dotted path (e.g., 'data.response.text') for nested fields. "
            "If None, operates on the value directly."
        ),
    )


__all__ = ["RedTeamingMiddlewareConfig"]

