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
"""Unit tests for the MLflow OTLP telemetry exporter routing header and config defaults."""

import base64

import nat.plugins.opentelemetry.register as otel_register


def test_mlflow_experiment_headers_match_mlflow_ingestion_contract():
    """The routing header key/value matches MLflow's OTLP ingestion (x-mlflow-experiment-id)."""
    assert otel_register._mlflow_experiment_headers("42") == {"x-mlflow-experiment-id": "42"}


def test_mlflow_exporter_config_field_defaults():
    """Defaults target a local MLflow tracking server and the default experiment."""
    fields = otel_register.MLflowTelemetryExporter.model_fields
    assert fields["endpoint"].default == "http://localhost:5000/v1/traces"
    assert fields["experiment_id"].default == "0"


def test_mlflow_exporter_config_auth_field_defaults():
    """Auth fields are optional and empty by default (unauthenticated local server)."""
    fields = otel_register.MLflowTelemetryExporter.model_fields
    assert fields["token"].default_factory() == otel_register.SerializableSecretStr("")
    assert fields["username"].default == ""
    assert fields["password"].default_factory() == otel_register.SerializableSecretStr("")


def test_mlflow_auth_headers_bearer_token():
    """A token produces a bearer Authorization header (mirrors MLFLOW_TRACKING_TOKEN)."""
    headers = otel_register._mlflow_auth_headers(token="tok_secret", username="user", password="pass")
    assert headers == {"Authorization": "Bearer tok_secret"}


def test_mlflow_auth_headers_basic_username_password():
    """Username/password produce a basic Authorization header (mirrors MLflow's client)."""
    expected = base64.b64encode(b"user:pass").decode("utf-8")
    headers = otel_register._mlflow_auth_headers(token=None, username="user", password="pass")
    assert headers == {"Authorization": f"Basic {expected}"}


def test_mlflow_auth_headers_token_takes_precedence_over_basic():
    """Token auth wins over basic auth, matching MLflow's own client precedence."""
    headers = otel_register._mlflow_auth_headers(token="tok_secret", username="user", password="pass")
    assert headers["Authorization"].startswith("Bearer ")


def test_mlflow_auth_headers_no_credentials_sends_no_authorization_header():
    """Without credentials no Authorization header is sent (unauthenticated local server)."""
    assert otel_register._mlflow_auth_headers(token=None, username="", password=None) == {}
    # Partial credentials cannot form basic auth either
    assert otel_register._mlflow_auth_headers(token=None, username="user", password=None) == {}
    assert otel_register._mlflow_auth_headers(token=None, username="", password="pass") == {}
