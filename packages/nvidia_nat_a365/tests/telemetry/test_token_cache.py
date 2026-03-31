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

"""Tests for _TokenCache class in A365 telemetry registration."""

from datetime import datetime, timedelta, timezone

import pytest

from nat.plugins.a365.telemetry.register import _TokenCache


class TestTokenCache:
    """Tests for _TokenCache thread-safe token caching."""

    def test_get_token_with_expiration_valid(self):
        """Test that valid token (not expiring soon) is returned."""
        expires_at = datetime.now(timezone.utc) + timedelta(minutes=10)
        cache = _TokenCache("test_token_123", expires_at)

        token = cache.get_token()
        assert token == "test_token_123"

    def test_get_token_with_expiration_expired(self):
        """Test that expired token returns None."""
        expires_at = datetime.now(timezone.utc) - timedelta(minutes=1)
        cache = _TokenCache("test_token_123", expires_at)

        token = cache.get_token()
        assert token is None

    def test_get_token_with_expiration_expiring_soon(self):
        """Test that token expiring within 5 minutes returns None."""
        expires_at = datetime.now(timezone.utc) + timedelta(minutes=3)
        cache = _TokenCache("test_token_123", expires_at)

        token = cache.get_token()
        assert token is None

    def test_get_token_without_expiration(self):
        """Test that token without expiration info is always returned."""
        cache = _TokenCache("test_token_123", None)

        token = cache.get_token()
        assert token == "test_token_123"

    def test_update_token(self):
        """Test updating token and expiration."""
        expires_at = datetime.now(timezone.utc) + timedelta(minutes=10)
        cache = _TokenCache("old_token", expires_at)

        new_expires_at = datetime.now(timezone.utc) + timedelta(minutes=20)
        cache.update_token("new_token", new_expires_at)

        assert cache.get_token() == "new_token"

    def test_is_expiring_soon_with_expiration(self):
        """Test is_expiring_soon returns True when token expires soon."""
        expires_at = datetime.now(timezone.utc) + timedelta(minutes=3)
        cache = _TokenCache("test_token", expires_at)

        assert cache.is_expiring_soon() is True

    def test_is_expiring_soon_not_expiring(self):
        """Test is_expiring_soon returns False when token is not expiring soon."""
        expires_at = datetime.now(timezone.utc) + timedelta(minutes=10)
        cache = _TokenCache("test_token", expires_at)

        assert cache.is_expiring_soon() is False

    def test_is_expiring_soon_without_expiration(self):
        """Test is_expiring_soon returns False when no expiration info."""
        cache = _TokenCache("test_token", None)

        assert cache.is_expiring_soon() is False
