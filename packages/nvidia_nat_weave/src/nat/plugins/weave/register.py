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

from nat.builder.builder import Builder
from nat.cli.register_workflow import register_telemetry_exporter
from nat.data_models.telemetry_exporter import TelemetryExporterBaseConfig

logger = logging.getLogger(__name__)


class WeaveTelemetryExporter(TelemetryExporterBaseConfig, name="weave"):
    """A telemetry exporter to transmit traces to Weights & Biases Weave using OpenTelemetry."""
    project: str = Field(description="The W&B project name.")
    entity: str | None = Field(default=None, description="The W&B username or team name.")
    redact_pii: bool = Field(default=False, description="Whether to redact PII from the traces.")
    redact_pii_fields: list[str] | None = Field(
        default=None,
        description="Custom list of PII entity types to redact. Only used when redact_pii=True. "
        "Examples: CREDIT_CARD, EMAIL_ADDRESS, PHONE_NUMBER, etc.")
    redact_keys: list[str] | None = Field(
        default=None,
        description="Additional keys to redact from traces beyond the default (api_key, auth_headers, authorization).")
    verbose: bool = Field(default=False, description="Whether to enable verbose logging.")


@register_telemetry_exporter(config_type=WeaveTelemetryExporter)
async def weave_telemetry_exporter(config: WeaveTelemetryExporter, builder: Builder):
    import weave
    from nat.plugins.weave.weave_exporter import WeaveExporter

    weave_settings = {}

    if config.redact_pii:
        weave_settings["redact_pii"] = True

        # Add custom fields if specified
        if config.redact_pii_fields:
            weave_settings["redact_pii_fields"] = config.redact_pii_fields

    project_name = f"{config.entity}/{config.project}" if config.entity else config.project

    if weave_settings:
        _ = weave.init(project_name=project_name, settings=weave_settings)
    else:
        _ = weave.init(project_name=project_name)

    # Handle custom redact keys if specified
    if config.redact_keys and config.redact_pii:
        # Need to create a new list combining default keys and custom ones
        from weave.trace import sanitize
        default_keys = sanitize.REDACT_KEYS

        # Create a new list with all keys
        all_keys = list(default_keys) + config.redact_keys

        # Replace the default REDACT_KEYS with our extended list
        sanitize.REDACT_KEYS = tuple(all_keys)

    yield WeaveExporter(project=config.project, entity=config.entity, verbose=config.verbose)
