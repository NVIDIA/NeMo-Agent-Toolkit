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

from unittest.mock import Mock
from unittest.mock import patch

import pytest

from nat.plugins.phoenix.mixin.phoenix_mixin import PhoenixMixin
from nat.plugins.phoenix.register import PhoenixTelemetryExporter
from nat.plugins.phoenix.register import _phoenix_auth_headers
from nat.plugins.phoenix.register import phoenix_telemetry_exporter


def test_phoenix_auth_headers_add_bearer_prefix():
    assert _phoenix_auth_headers("test-key") == {"authorization": "Bearer test-key"}


def test_phoenix_auth_headers_preserve_existing_bearer_prefix():
    assert _phoenix_auth_headers("Bearer test-key") == {"authorization": "Bearer test-key"}


class _BaseExporter:

    def __init__(self, resource_attributes=None):
        self.resource_attributes = resource_attributes


class _PhoenixMixinExporter(PhoenixMixin, _BaseExporter):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


@patch("nat.plugins.phoenix.mixin.phoenix_mixin.HTTPSpanExporter")
def test_phoenix_mixin_passes_headers_to_http_span_exporter(mock_http_span_exporter):
    headers = {"authorization": "Bearer test-key"}

    _PhoenixMixinExporter(endpoint="http://localhost:6006/v1/traces",
                          project="simple_calculator",
                          timeout=5.0,
                          headers=headers)

    mock_http_span_exporter.assert_called_once_with(endpoint="http://localhost:6006/v1/traces",
                                                    timeout=5.0,
                                                    headers=headers)


async def _collect_exporter(config: PhoenixTelemetryExporter):
    async with phoenix_telemetry_exporter(config, Mock()) as exporter:
        return exporter


@pytest.mark.usefixtures("restore_environ")
async def test_phoenix_telemetry_exporter_passes_api_key_header():
    config = PhoenixTelemetryExporter(endpoint="http://localhost:6006/v1/traces",
                                      project="simple_calculator",
                                      api_key="test-key")

    with patch("nat.plugins.phoenix.phoenix_exporter.PhoenixOtelExporter") as mock_exporter:
        await _collect_exporter(config)

    mock_exporter.assert_called_once()
    assert mock_exporter.call_args.kwargs["headers"] == {"authorization": "Bearer test-key"}


@pytest.mark.usefixtures("restore_environ")
async def test_phoenix_telemetry_exporter_uses_env_api_key(monkeypatch):
    monkeypatch.setenv("PHOENIX_API_KEY", "env-key")
    config = PhoenixTelemetryExporter(endpoint="http://localhost:6006/v1/traces", project="simple_calculator")

    with patch("nat.plugins.phoenix.phoenix_exporter.PhoenixOtelExporter") as mock_exporter:
        await _collect_exporter(config)

    mock_exporter.assert_called_once()
    assert mock_exporter.call_args.kwargs["headers"] == {"authorization": "Bearer env-key"}
