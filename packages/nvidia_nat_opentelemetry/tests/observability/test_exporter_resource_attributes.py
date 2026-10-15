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
"""The configured resource_attributes must reach the OTLP exporter."""

from unittest.mock import MagicMock
from unittest.mock import patch

import nat.plugins.opentelemetry.register as otel_register

RESOURCE_ATTRIBUTES = {"deployment.environment": "staging"}


async def _exported_resource_attributes(config, register_fn) -> dict[str, str]:
    with patch("nat.plugins.opentelemetry.OTLPSpanAdapterExporter") as mock_exporter:
        async with register_fn(config, MagicMock()):
            pass

    return mock_exporter.call_args.kwargs["resource_attributes"]


async def test_langfuse_exporter_passes_resource_attributes():
    config = otel_register.LangfuseTelemetryExporter(endpoint="http://localhost:3000/api/public/otel/v1/traces",
                                                     public_key="pk",
                                                     secret_key="sk",
                                                     resource_attributes=RESOURCE_ATTRIBUTES)

    exported = await _exported_resource_attributes(config, otel_register.langfuse_telemetry_exporter)

    assert exported == RESOURCE_ATTRIBUTES


async def test_langsmith_exporter_passes_resource_attributes():
    config = otel_register.LangsmithTelemetryExporter(project="demo",
                                                      api_key="key",
                                                      resource_attributes=RESOURCE_ATTRIBUTES)

    exported = await _exported_resource_attributes(config, otel_register.langsmith_telemetry_exporter)

    assert exported == RESOURCE_ATTRIBUTES


async def test_patronus_exporter_passes_resource_attributes():
    config = otel_register.PatronusTelemetryExporter(endpoint="https://otel.patronus.ai:4317",
                                                     project="demo",
                                                     api_key="key",
                                                     resource_attributes=RESOURCE_ATTRIBUTES)

    exported = await _exported_resource_attributes(config, otel_register.patronus_telemetry_exporter)

    assert exported == RESOURCE_ATTRIBUTES
