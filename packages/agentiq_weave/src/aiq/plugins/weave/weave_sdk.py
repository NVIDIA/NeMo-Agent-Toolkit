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

import os
from typing import Optional

from pydantic import Field

from aiq.builder.builder import Builder
from aiq.cli.register_workflow import register_telemetry_exporter
from aiq.data_models.telemetry_exporter import TelemetryExporterBaseConfig


def set_wandb_api_key(config_api_key: Optional[str] = None) -> Optional[str]:
    """
    Get the W&B API key from various sources in order of priority:
    1. Config provided key
    2. WANDB_API_KEY environment variable
    Returns:
        The API key if found, None otherwise
    """
    if config_api_key:
        return config_api_key
    # Check environment variable
    env_api_key = os.environ.get("WANDB_API_KEY")
    if env_api_key:
        return env_api_key
    return None


class WeaveTelemetryExporter(TelemetryExporterBaseConfig, name="weave"):
    """A telemetry exporter to transmit traces to Weights & Biases Weave using OpenTelemetry."""
    entity: str = Field(description="The W&B entity/organization.")
    project: str = Field(description="The W&B project name.")
    api_key: Optional[str] = Field(
        default=None,
        description="Your W&B API key for auth. If not provided, look for WANDB_API_KEY environment variable.")


@register_telemetry_exporter(config_type=WeaveTelemetryExporter)
async def weave_telemetry_exporter(config: WeaveTelemetryExporter, builder: Builder):
    if config.api_key:
        set_wandb_api_key(config.api_key)

    import weave
    _ = weave.init(project_name=f"{config.entity}/{config.project}")

    class NoOpSpanExporter:

        def export(self, spans):
            return None

        def shutdown(self):
            return None

    # just yielding None errors with 'NoneType' object has no attribute 'export'
    yield NoOpSpanExporter()
