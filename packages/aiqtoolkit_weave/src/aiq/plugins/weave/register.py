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

from pydantic import Field

from aiq.builder.builder import Builder
from aiq.cli.register_workflow import register_telemetry_exporter
from aiq.data_models.telemetry_exporter import TelemetryExporterBaseConfig

logger = logging.getLogger(__name__)


class WeaveTelemetryExporter(TelemetryExporterBaseConfig, name="new_weave"):
    """A telemetry exporter to transmit traces to Weights & Biases Weave using OpenTelemetry."""
    project: str = Field(description="The W&B project name.")
    entity: str | None = Field(default=None, description="The W&B username or team name.")


@register_telemetry_exporter(config_type=WeaveTelemetryExporter)
async def weave_telemetry_exporter(config: WeaveTelemetryExporter, builder: Builder):  # pylint: disable=unused-argument
    import weave

    from aiq.plugins.weave.weave_exporter import WeaveExporter

    if config.entity:
        _ = weave.init(project_name=f"{config.entity}/{config.project}")
    else:
        _ = weave.init(project_name=config.project)

    yield WeaveExporter(project=config.project, entity=config.entity)
