# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
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
"""Tests for _AgentTokenCache (per-(agent_id, tenant_id) token caching)."""

from datetime import UTC
from datetime import datetime
from datetime import timedelta

from nat.plugins.a365.telemetry.register import _AgentTokenCache

KEY_A = ("agent-A", "tenant-A")
KEY_B = ("agent-B", "tenant-B")
TOKEN_A = "token-A"
TOKEN_B = "token-B"


class TestAgentTokenCache:

    def test_get_token_returns_none_when_unset(self):
        cache = _AgentTokenCache()
        assert cache.get_token(*KEY_A) is None

    def test_update_then_get_returns_token(self):
        cache = _AgentTokenCache()
        expires = datetime.now(UTC) + timedelta(minutes=10)

        cache.update_token(*KEY_A, token=TOKEN_A, expires_at=expires)

        assert cache.get_token(*KEY_A) == TOKEN_A

    def test_get_returns_none_when_token_expired(self):
        cache = _AgentTokenCache()
        expires = datetime.now(UTC) - timedelta(minutes=1)

        cache.update_token(*KEY_A, token=TOKEN_A, expires_at=expires)

        assert cache.get_token(*KEY_A) is None

    def test_get_returns_none_when_token_expiring_soon(self):
        cache = _AgentTokenCache()
        expires = datetime.now(UTC) + timedelta(minutes=3)

        cache.update_token(*KEY_A, token=TOKEN_A, expires_at=expires)

        assert cache.get_token(*KEY_A) is None

    def test_get_returns_token_when_no_expiration_set(self):
        cache = _AgentTokenCache()

        cache.update_token(*KEY_A, token=TOKEN_A, expires_at=None)

        assert cache.get_token(*KEY_A) == TOKEN_A

    def test_keys_are_independent(self):
        cache = _AgentTokenCache()
        long = datetime.now(UTC) + timedelta(minutes=10)
        short = datetime.now(UTC) - timedelta(minutes=1)

        cache.update_token(*KEY_A, token=TOKEN_A, expires_at=long)
        cache.update_token(*KEY_B, token=TOKEN_B, expires_at=short)

        assert cache.get_token(*KEY_A) == TOKEN_A
        assert cache.get_token(*KEY_B) is None  # B is expired, A is not

    def test_is_expiring_soon_true_when_within_buffer(self):
        cache = _AgentTokenCache()
        expires = datetime.now(UTC) + timedelta(minutes=3)

        cache.update_token(*KEY_A, token=TOKEN_A, expires_at=expires)

        assert cache.is_expiring_soon(*KEY_A) is True

    def test_is_expiring_soon_false_when_outside_buffer(self):
        cache = _AgentTokenCache()
        expires = datetime.now(UTC) + timedelta(minutes=10)

        cache.update_token(*KEY_A, token=TOKEN_A, expires_at=expires)

        assert cache.is_expiring_soon(*KEY_A) is False

    def test_is_expiring_soon_false_when_no_expiration(self):
        cache = _AgentTokenCache()

        cache.update_token(*KEY_A, token=TOKEN_A, expires_at=None)

        assert cache.is_expiring_soon(*KEY_A) is False

    def test_is_expiring_soon_true_when_unset(self):
        """An unset key needs a fresh token, so 'expiring soon' should be True."""
        cache = _AgentTokenCache()

        assert cache.is_expiring_soon(*KEY_A) is True
