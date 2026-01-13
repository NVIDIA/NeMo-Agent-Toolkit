#
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
#

import logging
from typing import Any

from pydantic import ValidationError

from nat.data_models.api_server import ResponseIntermediateStep
from nat.data_models.intermediate_step import IntermediateStep
from nat.data_models.intermediate_step import IntermediateStepPayload
from nat.data_models.invocation_node import InvocationNode

logger = logging.getLogger(__name__)


def merge_trace_events(trace_payload: dict[str, Any] | None,
                       *,
                       default_parent: str = "remote",
                       default_function_name: str = "remote_function",
                       default_function_id: str = "remote_function_id") -> list[IntermediateStep]:
    """
    Convert child trace events into IntermediateStep objects.
    """
    merged: list[IntermediateStep] = []

    if not trace_payload or not isinstance(trace_payload, dict):
        return merged

    events = trace_payload.get("events") or []
    for event in events:
        try:
            resp_step = ResponseIntermediateStep.model_validate(event)
            payload = IntermediateStepPayload.model_validate_json(resp_step.payload)
            intermediate_step = IntermediateStep(
                parent_id=resp_step.parent_id or default_parent,
                function_ancestry=InvocationNode(function_name=payload.name or default_function_name,
                                                 function_id=payload.UUID or default_function_id),
                payload=payload,
            )
            merged.append(intermediate_step)
        except (ValidationError, Exception) as exc:  # broad by design to avoid breaking trace merge
            logger.exception("Failed to merge trace event: %s", exc)

    return merged
