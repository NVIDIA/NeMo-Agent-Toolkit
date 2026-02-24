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
"""Function group that exposes workspace actions as callable functions."""

from __future__ import annotations

import logging
import typing

from pydantic import BaseModel
from pydantic import Field
from pydantic import create_model

from nat.builder.function import FunctionGroup
from nat.cli.register_workflow import register_function_group
from nat.data_models.function import FunctionGroupBaseConfig
from nat.workspace.types import TypeSchema
from nat.workspace.types import WorkspaceActionSchema

if typing.TYPE_CHECKING:
    from nat.builder.builder import Builder

logger = logging.getLogger(__name__)

# Mapping from TypeSchema type strings to Python types.
_TYPE_MAP: dict[str, type] = {
    "string": str,
    "number": float,
    "integer": int,
    "boolean": bool,
    "object": dict,
    "array": list,
    "path": str,
    "content": str,
}


def _parse_param_name(ts: TypeSchema) -> tuple[str, str]:
    """Extract (param_name, description) from a TypeSchema.

    The convention is ``"param_name: Description text."``; if no colon is
    present the whole description is used as both name and description.
    """
    desc = ts.description
    if ":" in desc:
        name, _, remainder = desc.partition(":")
        return name.strip(), remainder.strip()
    return desc.strip(), desc.strip()


def _build_input_model(schema: WorkspaceActionSchema) -> type[BaseModel] | None:
    """Dynamically create a Pydantic model from a WorkspaceActionSchema's parameters."""
    if not schema.parameters:
        return None

    field_definitions: dict[str, typing.Any] = {}
    for ts in schema.parameters:
        param_name, description = _parse_param_name(ts)
        python_type = _TYPE_MAP.get(ts.type, typing.Any)
        field_definitions[param_name] = (
            python_type,
            Field(description=description),
        )

    model_name = f"{schema.name}_input"
    return create_model(model_name, **field_definitions)


class WorkspaceActionsConfig(FunctionGroupBaseConfig, name="workspace_actions"):
    """Configuration for exposing workspace actions as a function group.

    When used with the Builder system, workspace actions are automatically
    discovered from the configured workspace and registered as functions.
    """


@register_function_group(config_type=WorkspaceActionsConfig)
async def build_workspace_actions(config: WorkspaceActionsConfig, builder: Builder):
    """Build a function group from workspace actions.

    Discovers available actions from the configured workspace and exposes
    each as a callable function within the group.
    """
    from nat.data_models.workspace import ActionResult
    from nat.data_models.workspace import ActionStatus

    workspace_manager = await builder.get_workspace_manager()
    if workspace_manager is None:
        raise RuntimeError("No workspace configured. WorkspaceActionsConfig requires a workspace "
                           "to be configured in the workflow.")

    async with workspace_manager as workspace:
        action_schemas = await workspace.get_actions()

        group = FunctionGroup(config=config)

        for schema in action_schemas:
            ws = workspace
            input_model = _build_input_model(schema)

            # Capture loop variables in a factory to avoid late-binding issues.
            def _make_execute(action_name: str = schema.name, bound_ws=ws):

                async def _execute(**kwargs: typing.Any, ) -> str:
                    result: ActionResult = await bound_ws.execute_action(action_name, kwargs)
                    if result.status != ActionStatus.SUCCESS:
                        raise RuntimeError(result.error_message or f"Action failed with status {result.status}")
                    return str(result.output) if result.output is not None else ""

                return _execute

            group.add_function(
                name=schema.name,
                fn=_make_execute(),
                description=schema.description,
                input_schema=input_model,
            )

        logger.info("Registered %d workspace actions as functions", len(action_schemas))
        yield group
