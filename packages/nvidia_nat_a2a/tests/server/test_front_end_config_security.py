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

import logging

import pytest
from pydantic import ValidationError

from nat.plugins.a2a.server.front_end_config import A2AFrontEndConfig

# Bind addresses are meaningful in these tests, not accidental. A single
# module-level constant documents that intent and keeps Bandit's S104
# (bind all interfaces) suppression localized to one place — the test
# exists precisely to verify the behavior of binding "0.0.0.0" under
# various auth configurations.
WILDCARD_BIND = "0.0.0.0"  # noqa: S104
NAMED_HOST_BIND = "agent.example.com"


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
            A2AFrontEndConfig(host=WILDCARD_BIND)
        assert "without authentication" in str(exc_info.value)

    def test_non_localhost_bind_rejects_named_host(self):
        """A named host like 'agent.example.com' without server_auth also fails."""
        with pytest.raises(ValidationError) as exc_info:
            A2AFrontEndConfig(host=NAMED_HOST_BIND)
        assert "without authentication" in str(exc_info.value)

    def test_non_localhost_bind_allowed_with_explicit_ack(self, caplog):
        """Operators who run behind an external auth layer can opt in explicitly.

        This path must also emit a WARNING so operations teams still see the
        event in logs even though the config was intentional.
        """
        with caplog.at_level(logging.WARNING, logger="nat.plugins.a2a.server.front_end_config"):
            cfg = A2AFrontEndConfig(
                host=WILDCARD_BIND,
                allow_unauthenticated_network_bind=True,
            )
        assert cfg.host == WILDCARD_BIND
        assert cfg.allow_unauthenticated_network_bind is True
        assert cfg.server_auth is None
        # The validator must emit a warning when the opt-in is exercised so
        # operations still sees the event in logs.
        combined = caplog.text
        assert WILDCARD_BIND in combined
        assert "allow_unauthenticated_network_bind" in combined

    def test_non_localhost_bind_allowed_with_server_auth(self):
        """The primary intended path: non-localhost bind + configured server_auth = OK.

        Closes the authenticated branch of the validator matrix — if someone
        accidentally tightens the validator to reject server_auth too, this
        test catches it.
        """
        from nat.authentication.oauth2.oauth2_resource_server_config import OAuth2ResourceServerConfig
        oauth = OAuth2ResourceServerConfig(issuer_url="https://auth.example.com")
        cfg = A2AFrontEndConfig(host=WILDCARD_BIND, server_auth=oauth)
        assert cfg.host == WILDCARD_BIND
        assert cfg.server_auth is oauth
        # The opt-in flag should NOT be needed when server_auth is present.
        assert cfg.allow_unauthenticated_network_bind is False

    def test_allow_unauthenticated_network_bind_defaults_false(self):
        """The opt-in acknowledgement must be off by default."""
        cfg = A2AFrontEndConfig(host="localhost")
        assert cfg.allow_unauthenticated_network_bind is False
