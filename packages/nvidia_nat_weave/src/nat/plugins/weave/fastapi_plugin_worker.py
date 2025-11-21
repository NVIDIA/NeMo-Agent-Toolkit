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

import logging

from fastapi import FastAPI
from fastapi import HTTPException
from fastapi import Request

from nat.builder.workflow_builder import WorkflowBuilder
from nat.front_ends.fastapi.fastapi_front_end_plugin_worker import FastApiFrontEndPluginWorker
from nat.utils.type_utils import override

logger = logging.getLogger(__name__)


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
    async def add_routes(self, app: FastAPI, builder: WorkflowBuilder):
        """Add routes including Weave feedback endpoint if Weave is configured."""
        # Add all standard routes first
        await super().add_routes(app, builder)

        # Add Weave-specific routes
        await self._add_weave_feedback_route(app, builder)

    async def _add_weave_feedback_route(self, app: FastAPI, builder: WorkflowBuilder):
        """Add the Weave feedback endpoint if Weave telemetry is configured."""

        # Find Weave telemetry exporter configuration
        weave_config = None
        for exporter_config in builder._telemetry_exporters.values():
            if hasattr(exporter_config.config, 'project') and \
               exporter_config.config.__class__.__name__ == 'WeaveTelemetryExporter':
                weave_config = exporter_config.config
                break

        if not weave_config:
            logger.debug("Weave telemetry not configured, skipping feedback endpoint")
            return

        try:

            async def add_chat_feedback(request: Request, payload: dict):
                """Add reaction feedback for an assistant message via observability trace ID."""
                import weave

                # Get the weave project name from the configuration
                entity = weave_config.entity if hasattr(weave_config, 'entity') else None
                project = weave_config.project
                weave_project = f"{entity}/{project}" if entity else project

                # Extract parameters from payload
                observability_trace_id = payload.get('observability_trace_id')
                reaction_type = payload.get('reaction_type')

                if not observability_trace_id or not reaction_type:
                    raise HTTPException(status_code=400, detail="observability_trace_id and reaction_type are required")

                # Add feedback directly
                try:
                    client = weave.init(weave_project)
                    call = client.get_call(observability_trace_id)
                    call.feedback.add_reaction(reaction_type)
                    return {"message": f"Added reaction '{reaction_type}' to call {observability_trace_id}"}
                except Exception as e:
                    logger.error("Failed to add feedback to Weave: %s", e)
                    raise HTTPException(status_code=500, detail=f"Failed to add feedback: {str(e)}") from e

            app.add_api_route(
                path="/feedback",
                endpoint=add_chat_feedback,
                methods=["POST"],
                description="Set reaction feedback for an assistant message via observability trace ID",
                responses={
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
