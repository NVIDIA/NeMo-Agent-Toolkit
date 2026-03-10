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
    """BasicUserInfo derives a base64-encoded credential from username and password."""

    def test_credential_derived_from_username_password(self):
        """Input: username="alice", password="s3cret". Asserts credential == base64("alice:s3cret")."""
        info = BasicUserInfo(username="alice", password=SecretStr("s3cret"))
        expected: str = base64.b64encode(b"alice:s3cret").decode()
        assert info.credential == expected

    def test_password_is_secret(self):
        """Input: BasicUserInfo with password. Asserts password value is accessible but not in repr."""
        info = BasicUserInfo(username="alice", password=SecretStr("s3cret"))
        assert info.password.get_secret_value() == "s3cret"
        assert "s3cret" not in repr(info)

    def test_frozen(self):
        """Input: attempt to mutate username. Asserts raises ValidationError (model is frozen)."""
        info = BasicUserInfo(username="alice", password=SecretStr("s3cret"))
        with pytest.raises(ValidationError):
            info.username = "bob"

    def test_empty_username_rejected(self):
        """Input: empty username string. Asserts raises ValidationError from min_length=1."""
        with pytest.raises(ValidationError, match="String should have at least 1 character"):
            BasicUserInfo(username="", password=SecretStr("s3cret"))

    def test_extra_fields_forbidden(self):
        """Input: unexpected extra field. Asserts raises ValidationError (extra="forbid")."""
        with pytest.raises(ValidationError):
            BasicUserInfo(username="alice", password=SecretStr("s3cret"), extra="bad")

    def test_different_users_produce_different_credentials(self):
        """Input: two different username/password pairs. Asserts credentials differ."""
        a = BasicUserInfo(username="alice", password=SecretStr("pass1"))
        b = BasicUserInfo(username="bob", password=SecretStr("pass2"))
        assert a.credential != b.credential

    def test_same_input_produces_same_credential(self):
        """Input: identical username/password twice. Asserts credentials are equal."""
        a = BasicUserInfo(username="alice", password=SecretStr("pass"))
        b = BasicUserInfo(username="alice", password=SecretStr("pass"))
        assert a.credential == b.credential


class TestJwtUserInfo:
    """JwtUserInfo.identity_claim resolves the first non-empty value from email, preferred_username, sub."""

    def test_identity_claim_prefers_email(self):
        """Input: claims with email, preferred_username, sub. Asserts identity_claim == email."""
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
        """Input: claims with preferred_username and sub (no email). Asserts identity_claim == preferred_username."""
        info = JwtUserInfo(
            preferred_username="alice",
            subject="sub-123",
            claims={
                "preferred_username": "alice", "sub": "sub-123"
            },
        )
        assert info.identity_claim == "alice"

    def test_identity_claim_falls_back_to_sub(self):
        """Input: claims with only sub. Asserts identity_claim == sub."""
        info = JwtUserInfo(
            subject="sub-123",
            claims={"sub": "sub-123"},
        )
        assert info.identity_claim == "sub-123"

    def test_identity_claim_returns_none_when_empty(self):
        """Input: empty claims dict. Asserts identity_claim is None."""
        info = JwtUserInfo(claims={})
        assert info.identity_claim is None

    def test_identity_claim_ignores_whitespace_only(self):
        """Input: claims with whitespace-only email and sub. Asserts identity_claim is None."""
        info = JwtUserInfo(claims={"email": "  ", "sub": "  "})
        assert info.identity_claim is None

    def test_identity_claim_strips_whitespace(self):
        """Input: email claim with leading/trailing whitespace. Asserts identity_claim is trimmed."""
        info = JwtUserInfo(claims={"email": "  alice@example.com  "})
        assert info.identity_claim == "alice@example.com"

    def test_frozen(self):
        """Input: attempt to mutate email. Asserts raises ValidationError (model is frozen)."""
        info = JwtUserInfo(claims={"sub": "user1"})
        with pytest.raises(ValidationError):
            info.email = "new@example.com"


class TestUserInfoFromBasicUser:
    """UserInfo created from BasicUserInfo derives a deterministic UUID from the credential."""

    def test_get_user_id_returns_deterministic_uuid(self):
        """Input: BasicUserInfo("alice", "pass"). Asserts get_user_id() == uuid5(namespace, credential)."""
        info = UserInfo(basic_user=BasicUserInfo(username="alice", password=SecretStr("pass")))
        expected: str = str(uuid.uuid5(_USER_ID_NAMESPACE, info.basic_user.credential))
        assert info.get_user_id() == expected

    def test_same_basic_user_same_uuid(self):
        """Input: identical BasicUserInfo twice. Asserts both produce the same user_id."""
        a = UserInfo(basic_user=BasicUserInfo(username="alice", password=SecretStr("pass")))
        b = UserInfo(basic_user=BasicUserInfo(username="alice", password=SecretStr("pass")))
        assert a.get_user_id() == b.get_user_id()

    def test_different_basic_users_different_uuids(self):
        """Input: two different BasicUserInfo. Asserts different user_ids."""
        a = UserInfo(basic_user=BasicUserInfo(username="alice", password=SecretStr("pass")))
        b = UserInfo(basic_user=BasicUserInfo(username="bob", password=SecretStr("pass")))
        assert a.get_user_id() != b.get_user_id()

    def test_get_user_details_returns_basic_user(self):
        """Input: UserInfo with BasicUserInfo. Asserts get_user_details() returns the same BasicUserInfo instance."""
        basic = BasicUserInfo(username="alice", password=SecretStr("pass"))
        info = UserInfo(basic_user=basic)
        assert info.get_user_details() is basic

    def test_uuid_is_valid(self):
        """Input: UserInfo from BasicUserInfo. Asserts get_user_id() parses as a valid UUID v5."""
        info = UserInfo(basic_user=BasicUserInfo(username="alice", password=SecretStr("pass")))
        parsed: uuid.UUID = uuid.UUID(info.get_user_id())
        assert parsed.version == 5


class TestUserInfoFromApiKey:
    """UserInfo._from_api_key creates a user from an API key token string."""

    def test_deterministic_uuid_from_api_key(self):
        """Input: API key string. Asserts get_user_id() == uuid5(namespace, key)."""
        info: UserInfo = UserInfo._from_api_key("nvapi-abc123")
        expected: str = str(uuid.uuid5(_USER_ID_NAMESPACE, "nvapi-abc123"))
        assert info.get_user_id() == expected

    def test_same_key_same_uuid(self):
        """Input: same API key twice. Asserts both produce the same user_id."""
        a: UserInfo = UserInfo._from_api_key("nvapi-xyz")
        b: UserInfo = UserInfo._from_api_key("nvapi-xyz")
        assert a.get_user_id() == b.get_user_id()

    def test_different_keys_different_uuids(self):
        """Input: two different API keys. Asserts different user_ids."""
        a: UserInfo = UserInfo._from_api_key("key-a")
        b: UserInfo = UserInfo._from_api_key("key-b")
        assert a.get_user_id() != b.get_user_id()

    def test_get_user_details_returns_api_key_string(self):
        """Input: UserInfo from API key. Asserts get_user_details() returns the raw key string."""
        info: UserInfo = UserInfo._from_api_key("nvapi-my-key")
        assert info.get_user_details() == "nvapi-my-key"

    def test_api_key_uuid_matches_cookie_for_same_value(self):
        """Input: same string as API key and cookie. Asserts same user_id (shared namespace)."""
        api_info: UserInfo = UserInfo._from_api_key("same-value")
        cookie_info: UserInfo = UserInfo._from_session_cookie("same-value")
        assert api_info.get_user_id() == cookie_info.get_user_id()

    def test_uuid_is_valid(self):
        """Input: UserInfo from API key. Asserts get_user_id() parses as a valid UUID v5."""
        info: UserInfo = UserInfo._from_api_key("nvapi-test")
        parsed: uuid.UUID = uuid.UUID(info.get_user_id())
        assert parsed.version == 5


class TestUserInfoFromSessionCookie:
    """UserInfo._from_session_cookie creates a user from a session cookie value."""

    def test_deterministic_uuid_from_cookie(self):
        """Input: cookie string. Asserts get_user_id() == uuid5(namespace, cookie)."""
        info: UserInfo = UserInfo._from_session_cookie("abc123")
        expected: str = str(uuid.uuid5(_USER_ID_NAMESPACE, "abc123"))
        assert info.get_user_id() == expected

    def test_same_cookie_same_uuid(self):
        """Input: same cookie string twice. Asserts both produce the same user_id."""
        a: UserInfo = UserInfo._from_session_cookie("session-xyz")
        b: UserInfo = UserInfo._from_session_cookie("session-xyz")
        assert a.get_user_id() == b.get_user_id()

    def test_different_cookies_different_uuids(self):
        """Input: two different cookie strings. Asserts different user_ids."""
        a: UserInfo = UserInfo._from_session_cookie("cookie-a")
        b: UserInfo = UserInfo._from_session_cookie("cookie-b")
        assert a.get_user_id() != b.get_user_id()

    def test_get_user_details_returns_cookie_string(self):
        """Input: UserInfo from cookie. Asserts get_user_details() returns the raw cookie string."""
        info: UserInfo = UserInfo._from_session_cookie("my-cookie")
        assert info.get_user_details() == "my-cookie"


class TestUserInfoFromJwt:
    """UserInfo._from_jwt creates a user from a JwtUserInfo using identity_claim as the UUID source."""

    def _jwt_info(self, **overrides) -> JwtUserInfo:
        claims: dict = {"sub": "user-sub", **overrides}
        return JwtUserInfo(
            email=claims.get("email"),
            preferred_username=claims.get("preferred_username"),
            subject=claims.get("sub"),
            claims=claims,
        )

    def test_deterministic_uuid_from_jwt(self):
        """Input: JwtUserInfo with email. Asserts get_user_id() == uuid5(namespace, email)."""
        jwt_info: JwtUserInfo = self._jwt_info(email="alice@example.com")
        info: UserInfo = UserInfo._from_jwt(jwt_info)
        expected: str = str(uuid.uuid5(_USER_ID_NAMESPACE, "alice@example.com"))
        assert info.get_user_id() == expected

    def test_same_jwt_same_uuid(self):
        """Input: same JwtUserInfo twice. Asserts both produce the same user_id."""
        jwt_info: JwtUserInfo = self._jwt_info(sub="user-sub")
        a: UserInfo = UserInfo._from_jwt(jwt_info)
        b: UserInfo = UserInfo._from_jwt(jwt_info)
        assert a.get_user_id() == b.get_user_id()

    def test_different_identity_claims_different_uuids(self):
        """Input: two JwtUserInfos with different emails. Asserts different user_ids."""
        a: UserInfo = UserInfo._from_jwt(self._jwt_info(email="alice@example.com"))
        b: UserInfo = UserInfo._from_jwt(self._jwt_info(email="bob@example.com"))
        assert a.get_user_id() != b.get_user_id()

    def test_get_user_details_returns_jwt_info(self):
        """Input: UserInfo from JWT. Asserts get_user_details() returns the same JwtUserInfo instance."""
        jwt_info: JwtUserInfo = self._jwt_info(sub="user-sub")
        info: UserInfo = UserInfo._from_jwt(jwt_info)
        assert info.get_user_details() is jwt_info

    def test_raises_without_identity_claim(self):
        """Input: JwtUserInfo with empty claims. Asserts raises ValueError matching "no usable identity claim"."""
        jwt_info = JwtUserInfo(claims={})
        with pytest.raises(ValueError, match="no usable identity claim"):
            UserInfo._from_jwt(jwt_info)


class TestUserInfoNoSource:
    """UserInfo with no identity source returns empty user_id and None details."""

    def test_empty_user_info_has_empty_user_id(self):
        """Input: UserInfo(). Asserts get_user_id() == ""."""
        info = UserInfo()
        assert info.get_user_id() == ""

    def test_empty_user_info_details_none(self):
        """Input: UserInfo(). Asserts get_user_details() is None."""
        info = UserInfo()
        assert info.get_user_details() is None


class TestUserInfoCrossSourceUniqueness:
    """Different identity sources for the same raw value produce different user_ids where appropriate."""

    def test_cookie_vs_basic_user_different_uuids(self):
        """Input: cookie "alice" vs BasicUserInfo("alice", "alice"). Asserts different UUIDs."""
        cookie_info: UserInfo = UserInfo._from_session_cookie("alice")
        basic_info: UserInfo = UserInfo(basic_user=BasicUserInfo(username="alice", password=SecretStr("alice")))
        assert cookie_info.get_user_id() != basic_info.get_user_id()

    def test_cookie_vs_jwt_different_uuids(self):
        """Input: cookie "session-abc123" vs JWT with email. Asserts different UUIDs."""
        cookie_info: UserInfo = UserInfo._from_session_cookie("session-abc123")
        jwt_info: UserInfo = UserInfo._from_jwt(
            JwtUserInfo(email="alice@example.com", claims={"email": "alice@example.com"}))
        assert cookie_info.get_user_id() != jwt_info.get_user_id()


class TestConsoleRunUserCreation:
    """Console front-end creates a UserInfo via BasicUserInfo for ``nat run`` and ``nat eval``."""

    def test_console_run_user_produces_stable_id(self):
        """Input: nat_run_user BasicUserInfo created twice. Asserts same non-empty user_id both times."""
        id_1: str = UserInfo(basic_user=BasicUserInfo(username="nat_run_user",
                                                      password=SecretStr("nat_run_user")), ).get_user_id()
        id_2: str = UserInfo(basic_user=BasicUserInfo(username="nat_run_user",
                                                      password=SecretStr("nat_run_user")), ).get_user_id()
        assert isinstance(id_1, str)
        assert len(id_1) > 0
        assert id_1 == id_2

    def test_console_run_user_id_differs_from_eval_user(self):
        """Input: nat_run_user vs nat_eval_user. Asserts the two user_ids are different."""
        run_id: str = UserInfo(basic_user=BasicUserInfo(username="nat_run_user",
                                                        password=SecretStr("nat_run_user")), ).get_user_id()
        eval_id: str = UserInfo(basic_user=BasicUserInfo(username="nat_eval_user",
                                                         password=SecretStr("nat_eval_user")), ).get_user_id()
        assert run_id != eval_id


class TestYamlConfiguredUsers:
    """Config.users allows YAML-declared users whose UserInfo produces valid, stable user_ids."""

    def _make_config_with_users(self, users: dict[str, UserInfo]):
        from nat.data_models.config import Config
        return Config(users=users)

    def test_yaml_user_produces_non_empty_id(self):
        """Input: Config with one YAML user. Asserts get_user_id() returns a non-empty string."""
        config = self._make_config_with_users({
            "alice": UserInfo(basic_user=BasicUserInfo(username="alice", password=SecretStr("pw"))),
        })
        user_id: str = config.users["alice"].get_user_id()
        assert isinstance(user_id, str)
        assert len(user_id) > 0

    def test_yaml_user_id_is_deterministic(self):
        """Input: same Config.users constructed twice. Asserts identical user_ids."""
        config_1 = self._make_config_with_users({
            "alice": UserInfo(basic_user=BasicUserInfo(username="alice", password=SecretStr("pw"))),
        })
        config_2 = self._make_config_with_users({
            "alice": UserInfo(basic_user=BasicUserInfo(username="alice", password=SecretStr("pw"))),
        })
        assert config_1.users["alice"].get_user_id() == config_2.users["alice"].get_user_id()

    def test_yaml_different_users_get_different_ids(self):
        """Input: Config with alice and bob. Asserts their user_ids are different."""
        config = self._make_config_with_users({
            "alice": UserInfo(basic_user=BasicUserInfo(username="alice", password=SecretStr("alice_pw"))),
            "bob": UserInfo(basic_user=BasicUserInfo(username="bob", password=SecretStr("bob_pw"))),
        })
        assert config.users["alice"].get_user_id() != config.users["bob"].get_user_id()


class TestIdentityClaimEdgeCases:
    """identity_claim precedence when claim values are non-string or empty."""

    def test_identity_claim_non_string_value_skipped(self):
        """Input: claims with email=123 (int) and sub="user-1". Asserts identity_claim == "user-1"."""
        info = JwtUserInfo(claims={"email": 123, "sub": "user-1"})
        assert info.identity_claim == "user-1"

    def test_identity_claim_empty_string_skipped(self):
        """Input: claims with email="" and sub="user-1". Asserts identity_claim == "user-1"."""
        info = JwtUserInfo(claims={"email": "", "sub": "user-1"})
        assert info.identity_claim == "user-1"


class TestCredentialEncoding:
    """BasicUserInfo.credential encoding with special characters."""

    def test_credential_with_colon_in_password(self):
        """Input: password containing a colon. Asserts credential encodes correctly and user_id is non-empty."""
        info = BasicUserInfo(username="user", password=SecretStr("pa:ss"))
        expected: str = base64.b64encode(b"user:pa:ss").decode()
        assert info.credential == expected
        assert len(UserInfo(basic_user=info).get_user_id()) > 0

    def test_credential_with_unicode_characters(self):
        """Input: unicode username and password. Asserts deterministic user_id across two identical inputs."""
        a = UserInfo(basic_user=BasicUserInfo(username="用户", password=SecretStr("密码")))
        b = UserInfo(basic_user=BasicUserInfo(username="用户", password=SecretStr("密码")))
        expected: str = base64.b64encode("用户:密码".encode()).decode()
        assert a.basic_user.credential == expected
        assert a.get_user_id() == b.get_user_id()


class TestFromJwtFactoryDetails:
    """_from_jwt stores JwtUserInfo accessible via get_user_details."""

    def test_from_jwt_stores_jwt_info_accessible_via_get_user_details(self):
        """Input: JwtUserInfo with email. Asserts get_user_details() returns the JwtUserInfo with correct email."""
        jwt_info = JwtUserInfo(email="a@b.com", claims={"email": "a@b.com"})
        info: UserInfo = UserInfo._from_jwt(jwt_info)
        assert isinstance(info.get_user_details(), JwtUserInfo)
        assert info.get_user_details().email == "a@b.com"


class TestBasicUserPostInit:
    """BasicUserInfo post-init with special characters and minimum-length username."""

    def test_basic_user_with_special_chars_in_password(self):
        """Input: password with special chars. Asserts non-empty user_id and deterministic across two inputs."""
        a = UserInfo(basic_user=BasicUserInfo(username="u", password=SecretStr("p@:ss w0rd")))
        b = UserInfo(basic_user=BasicUserInfo(username="u", password=SecretStr("p@:ss w0rd")))
        assert len(a.get_user_id()) > 0
        assert a.get_user_id() == b.get_user_id()

    def test_basic_user_min_length_username_accepted(self):
        """Input: single-character username. Asserts no ValidationError and non-empty user_id."""
        info = UserInfo(basic_user=BasicUserInfo(username="a", password=SecretStr("p")))
        assert len(info.get_user_id()) > 0


class TestGetUserDetailsPrecedence:
    """get_user_details returns the highest-priority source when multiple are set."""

    def test_get_user_details_returns_api_key_not_cookie_when_both_set(self):
        """Input: UserInfo with both _api_key and _session_cookie set. Asserts returns api_key."""
        info = UserInfo()
        object.__setattr__(info, "_api_key", "key")
        object.__setattr__(info, "_session_cookie", "cookie")
        assert info.get_user_details() == "key"


class TestUserInfoFrozen:
    """UserInfo is frozen — no attribute mutation after construction."""

    def test_cannot_set_basic_user_after_creation(self):
        """Input: attempt to set basic_user on existing UserInfo. Asserts raises ValidationError."""
        info = UserInfo()
        with pytest.raises(ValidationError):
            info.basic_user = BasicUserInfo(username="alice", password=SecretStr("pass"))
