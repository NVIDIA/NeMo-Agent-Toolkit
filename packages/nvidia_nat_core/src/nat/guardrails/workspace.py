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
"""Workspace guardrail base types."""

from abc import ABC
from dataclasses import dataclass

from nat.data_models.workspace import ActionRequest
from nat.data_models.workspace import ActionResult


@dataclass(frozen=True)
class WorkspaceGuardrailViolation:
    """Guardrail failure details.

    Args:
        guardrail_name: Name of the guardrail that blocked the action.
        message: Description of the violation.
    """

    guardrail_name: str
    message: str


class WorkspaceGuardrail(ABC):
    """Base class for workspace guardrails."""

    name: str

    async def validate_action(self, action: ActionRequest) -> WorkspaceGuardrailViolation | None:
        """Validate an action before execution.

        Args:
            action: Structured action request.

        Returns:
            GuardrailViolation if blocked, otherwise None.
        """
        return None

    async def sanitize_result(self, action: ActionRequest, result: ActionResult) -> ActionResult:
        """Sanitize the action result before returning it to the model layer.

        Args:
            action: Structured action request.
            result: Action result to sanitize.

        Returns:
            ActionResult after any redaction or modification.
        """
        return result
