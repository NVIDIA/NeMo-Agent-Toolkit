# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from typing import Optional

from pydantic import Field

from aiq.builder.builder import Builder
from aiq.cli.register_workflow import register_telemetry_exporter
from aiq.data_models.telemetry_exporter import TelemetryExporterBaseConfig

logger = logging.getLogger(__name__)


class WeaveTelemetryExporter(TelemetryExporterBaseConfig, name="weave"):
    """A telemetry exporter to transmit traces to Weights & Biases Weave using OpenTelemetry."""
    project: str = Field(description="The W&B project name.")
    entity: Optional[str] = Field(default=None, description="The W&B username or team name.")


class NoOpSpanExporter:
    """A no-op span exporter that properly implements the SpanExporter interface."""

    def export(self, spans):
        """Export method that doesn't actually export spans."""
        return None

    def shutdown(self):
        """Shutdown method that cleans up any resources."""
        try:
            # Try to clean up weave client if it exists
            import weave
            if hasattr(weave, 'finish'):
                weave.finish()
        except Exception as e:
            logger.debug("Error shutting down weave client: %s", e)
        return None


@register_telemetry_exporter(config_type=WeaveTelemetryExporter)
async def weave_telemetry_exporter(config: WeaveTelemetryExporter, builder: Builder):
    import weave

    if config.entity:
        _ = weave.init(project_name=f"{config.entity}/{config.project}")
    else:
        _ = weave.init(project_name=config.project)

    yield NoOpSpanExporter()
