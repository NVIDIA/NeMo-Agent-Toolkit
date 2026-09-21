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
"""Unit tests for the MLflow OTLP telemetry exporter: routing header, auth headers, and config defaults."""

import base64

import pytest

import nat.plugins.opentelemetry.register as otel_register


def test_mlflow_experiment_headers_match_mlflow_ingestion_contract():
    """The routing header key/value matches MLflow's OTLP ingestion (x-mlflow-experiment-id)."""
    assert otel_register._mlflow_experiment_headers("42") == {"x-mlflow-experiment-id": "42"}


def test_mlflow_exporter_config_field_defaults():
    """Defaults target a local MLflow tracking server and the default experiment."""
    fields = otel_register.MLflowTelemetryExporter.model_fields
    assert fields["endpoint"].default == "http://localhost:5000/v1/traces"
    assert fields["experiment_id"].default == "0"
    assert fields["username"].default == ""
    assert fields["headers"].default_factory() == {}


def test_mlflow_auth_headers_bearer_token():
    """A token produces an Authorization: Bearer header."""
    assert otel_register._mlflow_auth_headers(token="abc", username="", password=None) == {
        "Authorization": "Bearer abc"
    }


def test_mlflow_auth_headers_basic():
    """Username and password produce a Basic Authorization header."""
    expected = base64.b64encode(b"user:pass").decode()
    assert otel_register._mlflow_auth_headers(token=None, username="user", password="pass") == {
        "Authorization": f"Basic {expected}"
    }


def test_mlflow_auth_headers_token_takes_precedence_over_basic():
    """Token wins over basic auth, matching the MLflow Python client."""
    headers = otel_register._mlflow_auth_headers(token="abc", username="user", password="pass")
    assert headers == {"Authorization": "Bearer abc"}


def test_mlflow_auth_headers_no_credentials():
    """No credentials means no Authorization header."""
    assert otel_register._mlflow_auth_headers(token=None, username="", password=None) == {}


def test_mlflow_auth_headers_incomplete_basic_raises():
    """A lone username or password is a configuration error."""
    with pytest.raises(ValueError, match="username and password"):
        otel_register._mlflow_auth_headers(token=None, username="user", password=None)
    with pytest.raises(ValueError, match="username and password"):
        otel_register._mlflow_auth_headers(token=None, username="", password="pass")


def test_parse_otel_env_headers(monkeypatch):
    """OTEL_EXPORTER_OTLP_HEADERS parses as comma-separated key=value pairs."""
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_HEADERS", "Authorization=Bearer env-token,x-key=value")
    assert otel_register._parse_otel_env_headers() == {
        "Authorization": "Bearer env-token",
        "x-key": "value",
    }


def test_parse_otel_env_headers_empty(monkeypatch):
    """An unset env var yields no headers."""
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_HEADERS", raising=False)
    assert otel_register._parse_otel_env_headers() == {}


def _build_mlflow_headers(monkeypatch, config_kwargs):
    """Run the mlflow exporter factory with a stubbed exporter and return the headers it used."""
    import nat.plugins.opentelemetry as otel_pkg

    created = []

    class StubExporter:

        def __init__(self, **kwargs):
            created.append(kwargs)

    monkeypatch.setattr(otel_pkg, "OTLPSpanAdapterExporter", StubExporter)
    config = otel_register.MLflowTelemetryExporter(**config_kwargs)
    return config, created


async def test_mlflow_factory_merges_headers_with_precedence(monkeypatch):
    """Env headers < config headers < auth config, and the experiment-id routing header always wins."""
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_HEADERS", "x-extra=env-value,Authorization=Bearer env-auth")
    config, created = _build_mlflow_headers(
        monkeypatch,
        {
            "experiment_id": "42",
            "token": "cfg-token",
            "headers": {
                "x-extra": "cfg-value"
            },
        },
    )
    async with otel_register.mlflow_telemetry_exporter(config, builder=None):
        pass
    headers = created[0]["headers"]
    assert headers["x-mlflow-experiment-id"] == "42"
    assert headers["Authorization"] == "Bearer cfg-token"
    assert headers["x-extra"] == "cfg-value"


async def test_mlflow_factory_env_fallbacks(monkeypatch):
    """Empty config fields fall back to MLFLOW_TRACKING_TOKEN/USERNAME/PASSWORD env vars."""
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_HEADERS", raising=False)
    monkeypatch.setenv("MLFLOW_TRACKING_TOKEN", "env-token")
    monkeypatch.setenv("MLFLOW_TRACKING_USERNAME", "env-user")
    monkeypatch.setenv("MLFLOW_TRACKING_PASSWORD", "env-pass")
    config, created = _build_mlflow_headers(monkeypatch, {})
    async with otel_register.mlflow_telemetry_exporter(config, builder=None):
        pass
    assert created[0]["headers"]["Authorization"] == "Bearer env-token"


async def test_mlflow_factory_incomplete_basic_auth_raises(monkeypatch):
    """The factory surfaces the incomplete basic-auth configuration error."""
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_HEADERS", raising=False)
    config, _ = _build_mlflow_headers(monkeypatch, {"username": "user"})
    with pytest.raises(ValueError, match="username and password"):
        async with otel_register.mlflow_telemetry_exporter(config, builder=None):
            pass
