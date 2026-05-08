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
"""Unit tests for Arize AX OTLP defaults and auth header mapping."""

import nat.plugins.opentelemetry.register as otel_register


def test_arize_ax_default_endpoints():
    """US and EU default OTLP URLs match arize-otel gRPC/HTTP host paths."""
    assert otel_register._arize_ax_default_endpoint(protocol="grpc", use_eu_region=False) == "https://otlp.arize.com/v1"
    assert (otel_register._arize_ax_default_endpoint(protocol="http",
                                                     use_eu_region=False) == "https://otlp.arize.com/v1/traces")
    assert (otel_register._arize_ax_default_endpoint(protocol="grpc",
                                                     use_eu_region=True) == "https://otlp.eu-west-1a.arize.com/v1")
    assert (otel_register._arize_ax_default_endpoint(
        protocol="http", use_eu_region=True) == "https://otlp.eu-west-1a.arize.com/v1/traces")


def test_arize_ax_auth_headers_match_arize_otel_convention():
    """OTLP metadata keys align with arize-otel (authorization, arize-space-id, etc.)."""
    h = otel_register._arize_ax_auth_headers(space_id="space-123", api_key="k_secret")
    assert h["authorization"] == "k_secret"
    assert h["api_key"] == "k_secret"
    assert h["arize-space-id"] == "space-123"
    assert h["space_id"] == "space-123"
    assert h["arize-interface"] == "otel"
