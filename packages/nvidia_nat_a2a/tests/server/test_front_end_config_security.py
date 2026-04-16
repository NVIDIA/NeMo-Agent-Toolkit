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
"""Tests for A2A front-end security configuration validation.

Covers the validator that rejects non-localhost binds without server_auth
unless the operator sets allow_unauthenticated_network_bind explicitly.
CWE-306 defense.
"""

import pytest
from pydantic import ValidationError

from nat.plugins.a2a.server.front_end_config import A2AFrontEndConfig


class TestNonLocalhostBindRequiresAuth:
    """Validator must reject binding to a non-localhost host without authentication."""

    @pytest.mark.parametrize("host", ["localhost", "127.0.0.1", "::1"])
    def test_localhost_bind_without_auth_is_allowed(self, host):
        """Localhost binds don't need server_auth — loopback is not network-accessible."""
        cfg = A2AFrontEndConfig(host=host)
        assert cfg.host == host
        assert cfg.server_auth is None

    def test_non_localhost_bind_without_auth_is_rejected(self):
        """0.0.0.0 (or any non-loopback) without server_auth must fail validation."""
        with pytest.raises(ValidationError) as exc_info:
            A2AFrontEndConfig(host="0.0.0.0")
        assert "without authentication" in str(exc_info.value)

    def test_non_localhost_bind_rejects_named_host(self):
        """A named host like 'agent.example.com' without server_auth also fails."""
        with pytest.raises(ValidationError) as exc_info:
            A2AFrontEndConfig(host="agent.example.com")
        assert "without authentication" in str(exc_info.value)

    def test_non_localhost_bind_allowed_with_explicit_ack(self):
        """Operators who run behind an external auth layer can opt in explicitly."""
        cfg = A2AFrontEndConfig(
            host="0.0.0.0",
            allow_unauthenticated_network_bind=True,
        )
        assert cfg.host == "0.0.0.0"
        assert cfg.allow_unauthenticated_network_bind is True
        assert cfg.server_auth is None

    def test_allow_unauthenticated_network_bind_defaults_false(self):
        """The opt-in acknowledgement must be off by default."""
        cfg = A2AFrontEndConfig(host="localhost")
        assert cfg.allow_unauthenticated_network_bind is False
