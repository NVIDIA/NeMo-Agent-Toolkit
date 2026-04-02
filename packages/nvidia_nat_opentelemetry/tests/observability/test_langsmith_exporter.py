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
"""Tests for LangsmithTelemetryExporter endpoint handling."""

from unittest.mock import MagicMock
from unittest.mock import patch

from nat.plugins.opentelemetry.register import LangsmithTelemetryExporter
from nat.plugins.opentelemetry.register import langsmith_telemetry_exporter


class TestLangsmithTelemetryExporterEndpoint:
    """Tests that the endpoint attribute of LangsmithTelemetryExporter is honoured."""

    DEFAULT_ENDPOINT = "https://api.smith.langchain.com/otel/v1/traces"
    CUSTOM_ENDPOINT = "https://custom.langsmith.example.com/otel/v1/traces"

    def test_config_stores_default_endpoint(self):
        """LangsmithTelemetryExporter uses the built-in default when no endpoint is supplied."""
        config = LangsmithTelemetryExporter(
            project="test-project",
            api_key="test-api-key",
        )
        assert config.endpoint == self.DEFAULT_ENDPOINT

    def test_config_stores_custom_endpoint(self):
        """LangsmithTelemetryExporter stores a user-supplied endpoint."""
        config = LangsmithTelemetryExporter(
            project="test-project",
            api_key="test-api-key",
            endpoint=self.CUSTOM_ENDPOINT,
        )
        assert config.endpoint == self.CUSTOM_ENDPOINT

    @patch("nat.plugins.opentelemetry.OTLPSpanAdapterExporter")
    async def test_exporter_function_uses_default_endpoint(self, mock_otlp_cls):
        """langsmith_telemetry_exporter passes the default endpoint to OTLPSpanAdapterExporter."""
        mock_otlp_cls.return_value = MagicMock()
        config = LangsmithTelemetryExporter(
            project="test-project",
            api_key="test-api-key",
        )
        builder = MagicMock()

        async with langsmith_telemetry_exporter(config, builder):
            pass

        _, kwargs = mock_otlp_cls.call_args
        assert kwargs["endpoint"] == self.DEFAULT_ENDPOINT

    @patch("nat.plugins.opentelemetry.OTLPSpanAdapterExporter")
    async def test_exporter_function_uses_custom_endpoint(self, mock_otlp_cls):
        """langsmith_telemetry_exporter passes the user-supplied endpoint to OTLPSpanAdapterExporter.

        This test is expected to fail until the bug where config.endpoint is ignored is fixed.
        """
        mock_otlp_cls.return_value = MagicMock()
        config = LangsmithTelemetryExporter(
            project="test-project",
            api_key="test-api-key",
            endpoint=self.CUSTOM_ENDPOINT,
        )
        builder = MagicMock()

        async with langsmith_telemetry_exporter(config, builder):
            pass

        _, kwargs = mock_otlp_cls.call_args
        assert kwargs["endpoint"] == self.CUSTOM_ENDPOINT, (
            f"Expected endpoint {self.CUSTOM_ENDPOINT!r} but got {kwargs['endpoint']!r}. "
            "The config endpoint is being ignored."
        )
