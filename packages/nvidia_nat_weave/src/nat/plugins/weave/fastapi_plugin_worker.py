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

import asyncio
import logging

from fastapi import FastAPI
from fastapi import HTTPException
from fastapi import Request
from pydantic import BaseModel
from pydantic import field_validator
from pydantic import model_validator

from nat.builder.workflow_builder import WorkflowBuilder
from nat.front_ends.fastapi.fastapi_front_end_plugin_worker import FastApiFrontEndPluginWorker
from nat.runtime.session import SessionManager
from nat.utils.type_utils import override

logger = logging.getLogger(__name__)


class WeaveFeedbackPayload(BaseModel):
    """Payload for adding feedback to a Weave trace."""

    observability_trace_id: str
    reaction_type: str | None = None
    comment: str | None = None

    @field_validator('comment')
    @classmethod
    def validate_comment_length(cls, v: str | None) -> str | None:
        """Validate that comment does not exceed Weave's 1024 character limit."""
        if v is not None and len(v) > 1024:
            raise ValueError('Comment must not exceed 1024 characters')
        return v

    @model_validator(mode='after')
    def validate_at_least_one_feedback(self) -> 'WeaveFeedbackPayload':
        """Validate that at least one feedback field is provided."""
        if not self.reaction_type and not self.comment:
            raise ValueError("At least one of 'reaction_type' or 'comment' must be provided")
        return self


class WeaveFeedbackResponse(BaseModel):
    """Response for feedback submission."""

    message: str


class WeaveFastAPIPluginWorker(FastApiFrontEndPluginWorker):
    """FastAPI plugin worker that adds Weave-specific routes.

    This worker extends the default FastAPI worker to automatically add
    Weave feedback endpoints when Weave telemetry is configured.

    Usage:
        Configure your workflow to use this worker:

        .. code-block:: yaml

            general:
              front_end:
                _type: fastapi
                runner_class: nat.plugins.weave.fastapi_plugin_worker.WeaveFastAPIPluginWorker
    """

    @override
    async def add_routes(self, app: FastAPI, builder: WorkflowBuilder) -> None:
        """Add routes including Weave feedback endpoint if Weave is configured."""
        # Add all standard routes first
        await super().add_routes(app, builder)

        # Add Weave-specific routes
        await self._add_weave_feedback_route(app, builder)

    async def _add_weave_feedback_route(self, app: FastAPI, builder: WorkflowBuilder) -> None:
        """Add the Weave feedback endpoint if Weave telemetry is configured."""

        # Find Weave telemetry exporter configuration
        weave_config = None
        for exporter_config in builder._telemetry_exporters.values():
            if exporter_config.config.__class__.__name__ == 'WeaveTelemetryExporter':
                weave_config = exporter_config.config
                break

        if not weave_config:
            logger.debug("Weave telemetry not configured, skipping feedback endpoint")
            return

        try:
            session_manager = await SessionManager.create(config=self._config, shared_builder=builder)

            # Get the weave project name from the configuration
            entity = weave_config.entity
            project = weave_config.project
            weave_project = f"{entity}/{project}" if entity else project

            async def add_chat_feedback(request: Request, payload: WeaveFeedbackPayload) -> WeaveFeedbackResponse:
                """Add reaction and/or comment feedback for an assistant message via observability trace ID."""

                async with session_manager.session(http_connection=request,
                                                   user_authentication_callback=self._http_flow_handler.authenticate):
                    observability_trace_id = payload.observability_trace_id
                    reaction_type = payload.reaction_type
                    comment = payload.comment

                    def add_weave_feedback():
                        import weave

                        client = weave.init(weave_project)
                        call = client.get_call(observability_trace_id)

                        feedback_added = []
                        if reaction_type:
                            call.feedback.add_reaction(reaction_type)
                            feedback_added.append(f"reaction '{reaction_type}'")

                        if comment:
                            call.feedback.add_note(comment)
                            feedback_added.append("comment")

                        return feedback_added

                    try:
                        feedback_added = await asyncio.to_thread(add_weave_feedback)
                        feedback_str = " and ".join(feedback_added)
                        return WeaveFeedbackResponse(message=f"Added {feedback_str} to call {observability_trace_id}")
                    except Exception as e:
                        logger.error("Failed to add feedback to Weave: %s", e)
                        raise HTTPException(status_code=500, detail=f"Failed to add feedback: {str(e)}") from e

            app.add_api_route(
                path="/feedback",
                endpoint=add_chat_feedback,
                methods=["POST"],
                description=(
                    "Add reaction and/or comment feedback for an assistant message via observability trace ID. "
                    "Comments are limited to 1024 characters."),
                responses={
                    422: {
                        "description": "Validation Error - Invalid input",
                        "content": {
                            "application/json": {
                                "examples": {
                                    "missing_feedback": {
                                        "summary": "Missing required feedback",
                                        "value": {
                                            "detail": [{
                                                "type": "value_error",
                                                "loc": ["body"],
                                                "msg": "At least one of 'reaction_type' or 'comment' must be provided"
                                            }]
                                        }
                                    },
                                    "comment_too_long": {
                                        "summary": "Comment exceeds length limit",
                                        "value": {
                                            "detail": [{
                                                "type": "value_error",
                                                "loc": ["body", "comment"],
                                                "msg": "Comment must not exceed 1024 characters"
                                            }]
                                        }
                                    }
                                }
                            }
                        },
                    },
                    500: {
                        "description": "Internal Server Error",
                        "content": {
                            "application/json": {
                                "example": {
                                    "detail": "Internal server error occurred"
                                }
                            }
                        },
                    }
                },
            )

            logger.info("Registered Weave feedback endpoint at /feedback")

        except Exception as e:
            logger.warning("Failed to register Weave feedback endpoint: %s", e)
