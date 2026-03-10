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
"""Tests for UserInfo, JwtUserInfo, and BasicUserInfo data models."""

import base64
import uuid

import pytest
from pydantic import SecretStr
from pydantic import ValidationError

from nat.data_models.user_info import _USER_ID_NAMESPACE
from nat.data_models.user_info import BasicUserInfo
from nat.data_models.user_info import JwtUserInfo
from nat.data_models.user_info import UserInfo


class TestBasicUserInfo:

    def test_credential_derived_from_username_password(self):
        info = BasicUserInfo(username="alice", password=SecretStr("s3cret"))
        expected: str = base64.b64encode(b"alice:s3cret").decode()
        assert info.credential == expected

    def test_password_is_secret(self):
        info = BasicUserInfo(username="alice", password=SecretStr("s3cret"))
        assert info.password.get_secret_value() == "s3cret"
        assert "s3cret" not in repr(info)

    def test_frozen(self):
        info = BasicUserInfo(username="alice", password=SecretStr("s3cret"))
        with pytest.raises(ValidationError):
            info.username = "bob"

    def test_extra_fields_forbidden(self):
        with pytest.raises(ValidationError):
            BasicUserInfo(username="alice", password=SecretStr("s3cret"), extra="bad")

    def test_different_users_produce_different_credentials(self):
        a = BasicUserInfo(username="alice", password=SecretStr("pass1"))
        b = BasicUserInfo(username="bob", password=SecretStr("pass2"))
        assert a.credential != b.credential

    def test_same_input_produces_same_credential(self):
        a = BasicUserInfo(username="alice", password=SecretStr("pass"))
        b = BasicUserInfo(username="alice", password=SecretStr("pass"))
        assert a.credential == b.credential


class TestJwtUserInfo:

    def test_identity_claim_prefers_email(self):
        info = JwtUserInfo(
            email="alice@example.com",
            preferred_username="alice",
            subject="sub-123",
            claims={
                "email": "alice@example.com", "preferred_username": "alice", "sub": "sub-123"
            },
        )
        assert info.identity_claim == "alice@example.com"

    def test_identity_claim_falls_back_to_preferred_username(self):
        info = JwtUserInfo(
            preferred_username="alice",
            subject="sub-123",
            claims={
                "preferred_username": "alice", "sub": "sub-123"
            },
        )
        assert info.identity_claim == "alice"

    def test_identity_claim_falls_back_to_sub(self):
        info = JwtUserInfo(
            subject="sub-123",
            claims={"sub": "sub-123"},
        )
        assert info.identity_claim == "sub-123"

    def test_identity_claim_returns_none_when_empty(self):
        info = JwtUserInfo(claims={})
        assert info.identity_claim is None

    def test_identity_claim_ignores_whitespace_only(self):
        info = JwtUserInfo(claims={"email": "  ", "sub": "  "})
        assert info.identity_claim is None

    def test_identity_claim_strips_whitespace(self):
        info = JwtUserInfo(claims={"email": "  alice@example.com  "})
        assert info.identity_claim == "alice@example.com"

    def test_frozen(self):
        info = JwtUserInfo(claims={"sub": "user1"})
        with pytest.raises(ValidationError):
            info.email = "new@example.com"


class TestUserInfoFromBasicUser:

    def test_get_user_id_returns_deterministic_uuid(self):
        info = UserInfo(basic_user=BasicUserInfo(username="alice", password=SecretStr("pass")))
        expected: str = str(uuid.uuid5(_USER_ID_NAMESPACE, info.basic_user.credential))
        assert info.get_user_id() == expected

    def test_same_basic_user_same_uuid(self):
        a = UserInfo(basic_user=BasicUserInfo(username="alice", password=SecretStr("pass")))
        b = UserInfo(basic_user=BasicUserInfo(username="alice", password=SecretStr("pass")))
        assert a.get_user_id() == b.get_user_id()

    def test_different_basic_users_different_uuids(self):
        a = UserInfo(basic_user=BasicUserInfo(username="alice", password=SecretStr("pass")))
        b = UserInfo(basic_user=BasicUserInfo(username="bob", password=SecretStr("pass")))
        assert a.get_user_id() != b.get_user_id()

    def test_get_user_details_returns_basic_user(self):
        basic = BasicUserInfo(username="alice", password=SecretStr("pass"))
        info = UserInfo(basic_user=basic)
        assert info.get_user_details() is basic

    def test_uuid_is_valid(self):
        info = UserInfo(basic_user=BasicUserInfo(username="alice", password=SecretStr("pass")))
        parsed: uuid.UUID = uuid.UUID(info.get_user_id())
        assert parsed.version == 5


class TestUserInfoFromSessionCookie:

    def test_deterministic_uuid_from_cookie(self):
        info: UserInfo = UserInfo._from_session_cookie("abc123")
        expected: str = str(uuid.uuid5(_USER_ID_NAMESPACE, "abc123"))
        assert info.get_user_id() == expected

    def test_same_cookie_same_uuid(self):
        a: UserInfo = UserInfo._from_session_cookie("session-xyz")
        b: UserInfo = UserInfo._from_session_cookie("session-xyz")
        assert a.get_user_id() == b.get_user_id()

    def test_different_cookies_different_uuids(self):
        a: UserInfo = UserInfo._from_session_cookie("cookie-a")
        b: UserInfo = UserInfo._from_session_cookie("cookie-b")
        assert a.get_user_id() != b.get_user_id()

    def test_get_user_details_returns_cookie_string(self):
        info: UserInfo = UserInfo._from_session_cookie("my-cookie")
        assert info.get_user_details() == "my-cookie"


class TestUserInfoFromJwt:

    def _jwt_info(self, **overrides) -> JwtUserInfo:
        claims: dict = {"sub": "user-sub", **overrides}
        return JwtUserInfo(
            email=claims.get("email"),
            preferred_username=claims.get("preferred_username"),
            subject=claims.get("sub"),
            claims=claims,
        )

    def test_deterministic_uuid_from_jwt(self):
        jwt_info: JwtUserInfo = self._jwt_info(email="alice@example.com")
        info: UserInfo = UserInfo._from_jwt(jwt_info)
        expected: str = str(uuid.uuid5(_USER_ID_NAMESPACE, "alice@example.com"))
        assert info.get_user_id() == expected

    def test_same_jwt_same_uuid(self):
        jwt_info: JwtUserInfo = self._jwt_info(sub="user-sub")
        a: UserInfo = UserInfo._from_jwt(jwt_info)
        b: UserInfo = UserInfo._from_jwt(jwt_info)
        assert a.get_user_id() == b.get_user_id()

    def test_different_identity_claims_different_uuids(self):
        a: UserInfo = UserInfo._from_jwt(self._jwt_info(email="alice@example.com"))
        b: UserInfo = UserInfo._from_jwt(self._jwt_info(email="bob@example.com"))
        assert a.get_user_id() != b.get_user_id()

    def test_get_user_details_returns_jwt_info(self):
        jwt_info: JwtUserInfo = self._jwt_info(sub="user-sub")
        info: UserInfo = UserInfo._from_jwt(jwt_info)
        assert info.get_user_details() is jwt_info

    def test_raises_without_identity_claim(self):
        jwt_info = JwtUserInfo(claims={})
        with pytest.raises(ValueError, match="no usable identity claim"):
            UserInfo._from_jwt(jwt_info)


class TestUserInfoNoSource:

    def test_empty_user_info_has_empty_user_id(self):
        info = UserInfo()
        assert info.get_user_id() == ""

    def test_empty_user_info_details_none(self):
        info = UserInfo()
        assert info.get_user_details() is None


class TestUserInfoCrossSourceUniqueness:

    def test_cookie_vs_basic_user_different_uuids(self):
        cookie_info: UserInfo = UserInfo._from_session_cookie("alice")
        basic_info: UserInfo = UserInfo(basic_user=BasicUserInfo(username="alice", password=SecretStr("alice")))
        assert cookie_info.get_user_id() != basic_info.get_user_id()

    def test_cookie_vs_jwt_different_uuids(self):
        """A session cookie value and a JWT with a different identity claim produce different UUIDs."""
        cookie_info: UserInfo = UserInfo._from_session_cookie("session-abc123")
        jwt_info: UserInfo = UserInfo._from_jwt(
            JwtUserInfo(email="alice@example.com", claims={"email": "alice@example.com"}))
        assert cookie_info.get_user_id() != jwt_info.get_user_id()


class TestUserInfoFrozen:

    def test_cannot_set_basic_user_after_creation(self):
        info = UserInfo()
        with pytest.raises(ValidationError):
            info.basic_user = BasicUserInfo(username="alice", password=SecretStr("pass"))
