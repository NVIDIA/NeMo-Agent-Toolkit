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
"""Tests for A365 per-turn identity contextvar and activity extraction."""

import logging
from types import SimpleNamespace

from nat.plugins.a365.turn_context import A365TurnIdentity
from nat.plugins.a365.turn_context import extract_identity_from_activity
from nat.plugins.a365.turn_context import get_turn_identity
from nat.plugins.a365.turn_context import set_turn_identity


def _activity_with_methods(*, app_id, tenant, user, agentic=True):
    """Build an object that quacks like microsoft_agents.activity.Activity."""
    obj = SimpleNamespace()
    obj.is_agentic_request = lambda: agentic
    obj.get_agentic_instance_id = lambda: app_id
    obj.get_agentic_tenant_id = lambda: tenant
    obj.get_agentic_user = lambda: user
    return obj


def _activity_field_shaped(*, role, app_id, tenant, user):
    """Build a fallback shape: recipient with role + agentic_app_id + tenant_id."""
    recipient = SimpleNamespace(
        role=role,
        agentic_app_id=app_id,
        agentic_user_id=user,
        tenant_id=tenant,
    )
    return SimpleNamespace(recipient=recipient, conversation=None)


class TestExtractIdentityFromActivity:

    def test_uses_sdk_methods_when_available(self):
        activity = _activity_with_methods(app_id="agent-A", tenant="tenant-A", user="user-1")

        identity = extract_identity_from_activity(activity)

        assert identity == A365TurnIdentity(
            agent_app_id="agent-A",
            tenant_id="tenant-A",
            on_behalf_user_id="user-1",
        )

    def test_returns_none_when_not_agentic(self):
        activity = _activity_with_methods(app_id=None, tenant=None, user=None, agentic=False)

        assert extract_identity_from_activity(activity) is None

    def test_falls_back_to_recipient_fields(self):
        activity = _activity_field_shaped(
            role="agenticIdentity",
            app_id="agent-B",
            tenant="tenant-B",
            user="user-2",
        )

        identity = extract_identity_from_activity(activity)

        assert identity == A365TurnIdentity(
            agent_app_id="agent-B",
            tenant_id="tenant-B",
            on_behalf_user_id="user-2",
        )

    def test_field_shape_non_agentic_role_returns_none(self):
        activity = _activity_field_shaped(role="user", app_id="x", tenant="y", user="z")

        assert extract_identity_from_activity(activity) is None

    def test_returns_none_when_app_id_missing(self):
        """Without an agent app id we cannot route telemetry; treat as no identity."""
        activity = _activity_with_methods(app_id=None, tenant="tenant-A", user="user-1")

        assert extract_identity_from_activity(activity) is None

    def test_handles_none_activity(self):
        assert extract_identity_from_activity(None) is None


class TestSetTurnIdentity:

    def test_context_manager_sets_and_resets(self):
        assert get_turn_identity() is None

        identity = A365TurnIdentity(
            agent_app_id="agent-A",
            tenant_id="tenant-A",
            on_behalf_user_id="user-1",
        )

        with set_turn_identity(identity):
            assert get_turn_identity() == identity

        assert get_turn_identity() is None

    def test_nested_context_managers_restore_outer(self):
        outer = A365TurnIdentity("agent-A", "tenant-A", "user-1")
        inner = A365TurnIdentity("agent-B", "tenant-B", "user-2")

        with set_turn_identity(outer):
            assert get_turn_identity() == outer
            with set_turn_identity(inner):
                assert get_turn_identity() == inner
            assert get_turn_identity() == outer

        assert get_turn_identity() is None

    def test_set_turn_identity_accepts_none(self):
        outer = A365TurnIdentity("agent-A", "tenant-A", None)

        with set_turn_identity(outer):
            with set_turn_identity(None):
                assert get_turn_identity() is None
            assert get_turn_identity() == outer


# -----------------------------------------------------------------------------
# Tests targeting fields and code paths the original suite did not cover.
# These exercise the SDK-shaped attributes downstream telemetry actually
# consumes (channel name, conversation id, service URL, caller info, client
# address), plus the fallback chains and exception paths.
# -----------------------------------------------------------------------------


def _agentic_activity(
    *,
    app_id="agent-X",
    tenant="tenant-X",
    user="user-X",
    conversation=None,
    channel_id=None,
    service_url=None,
    from_property=None,
    channel_data=None,
    recipient_overrides=None,
):
    """Build a fully-shaped agentic activity for end-to-end identity extraction tests.

    All optional bits default to None / unset so tests can exercise one field at a time.
    """
    recipient = SimpleNamespace(role="agenticIdentity")
    if recipient_overrides:
        for key, value in recipient_overrides.items():
            setattr(recipient, key, value)

    activity = SimpleNamespace(
        is_agentic_request=lambda: True,
        get_agentic_instance_id=lambda: app_id,
        get_agentic_tenant_id=lambda: tenant,
        get_agentic_user=lambda: user,
        recipient=recipient,
        conversation=conversation,
        channel_id=channel_id,
        service_url=service_url,
        from_property=from_property,
        channel_data=channel_data,
    )
    return activity


class TestChannelNameNormalization:
    """``_normalize_channel_name`` should collapse Teams variants to canonical ``msteams``."""

    def test_canonical_msteams_passes_through(self):
        activity = _agentic_activity(channel_id="msteams")
        identity = extract_identity_from_activity(activity)
        assert identity.channel_name == "msteams"

    def test_legacy_teams_normalizes_to_msteams(self):
        activity = _agentic_activity(channel_id="Teams")
        identity = extract_identity_from_activity(activity)
        assert identity.channel_name == "msteams"

    def test_ms_teams_with_hyphen_normalizes(self):
        activity = _agentic_activity(channel_id="ms-teams")
        identity = extract_identity_from_activity(activity)
        assert identity.channel_name == "msteams"

    def test_non_teams_channel_passes_through_unchanged(self):
        activity = _agentic_activity(channel_id="webchat")
        identity = extract_identity_from_activity(activity)
        assert identity.channel_name == "webchat"

    def test_missing_channel_id_yields_none(self):
        activity = _agentic_activity(channel_id=None)
        identity = extract_identity_from_activity(activity)
        assert identity.channel_name is None


class TestConversationAndServiceUrlExtraction:
    """Conversation id and service URL are extracted when present."""

    def test_conversation_id_extracted(self):
        conversation = SimpleNamespace(id="conv-123")
        activity = _agentic_activity(conversation=conversation)
        identity = extract_identity_from_activity(activity)
        assert identity.conversation_id == "conv-123"

    def test_conversation_id_empty_string_becomes_none(self):
        """``_as_nonempty_str`` should reject empty strings."""
        conversation = SimpleNamespace(id="")
        activity = _agentic_activity(conversation=conversation)
        identity = extract_identity_from_activity(activity)
        assert identity.conversation_id is None

    def test_service_url_extracted_via_snake_case(self):
        activity = _agentic_activity(service_url="https://smba.example/foo")
        identity = extract_identity_from_activity(activity)
        assert identity.service_url == "https://smba.example/foo"

    def test_service_url_falls_back_to_camel_case(self):
        """SDK uses snake_case but JSON payloads use ``serviceUrl``; either is accepted."""
        activity = _agentic_activity()
        # Override: drop snake_case, add camelCase.
        activity.service_url = None
        activity.serviceUrl = "https://camel.example/x"
        identity = extract_identity_from_activity(activity)
        assert identity.service_url == "https://camel.example/x"


class TestCallerFromPropertyExtraction:
    """``from_property`` (the SDK alias for the JSON ``"from"`` field) is the caller source."""

    def test_caller_aad_object_id_preferred_over_id(self):
        from_property = SimpleNamespace(
            id="ch:12345",
            aad_object_id="aad-oid-7777",
            name="Alice Example",
        )
        activity = _agentic_activity(from_property=from_property)
        identity = extract_identity_from_activity(activity)
        assert identity.user_id == "aad-oid-7777"  # not "ch:12345"
        assert identity.user_name == "Alice Example"

    def test_caller_falls_back_to_id_when_no_aad_oid(self):
        from_property = SimpleNamespace(id="ch:12345", name="Bob")
        activity = _agentic_activity(from_property=from_property)
        identity = extract_identity_from_activity(activity)
        assert identity.user_id == "ch:12345"

    def test_caller_email_field_picks_first_present(self):
        from_property = SimpleNamespace(
            id="ch:1",
            user_principal_name="upn@example.com",
        )
        activity = _agentic_activity(from_property=from_property)
        identity = extract_identity_from_activity(activity)
        assert identity.user_email == "upn@example.com"

    def test_caller_none_yields_none_fields(self):
        """When the activity has no ``from_property``, caller fields are None."""
        activity = _agentic_activity(from_property=None)
        identity = extract_identity_from_activity(activity)
        assert identity.user_id is None
        assert identity.user_name is None
        assert identity.user_email is None

    def test_caller_does_not_resolve_via_pydantic_from_alias(self):
        """M15 regression: the dead ``"from"`` fallback was removed.

        Setting an attribute literally named ``"from"`` (legal via ``setattr`` but never
        produced by SDK Pydantic models, since ``alias="from"`` only affects (de)serialization)
        must NOT be picked up. Production behavior is to use ``from_property`` exclusively.
        """
        activity = _agentic_activity(from_property=None)
        # Force a literal "from" attribute -- a hand-built duck-type that the dead fallback
        # used to honor. After M15 this is ignored.
        setattr(activity, "from", SimpleNamespace(id="ch:should-be-ignored"))
        identity = extract_identity_from_activity(activity)
        assert identity.user_id is None, (
            "Regression: the dead `from` fallback was reintroduced; production SDK objects "
            "never expose `from` via attribute access (Pydantic alias only affects JSON).")


class TestClientAddressExtraction:
    """``_extract_client_address`` walks a chain of channel_data key variants."""

    def test_client_address_from_dict_channel_data(self):
        activity = _agentic_activity(channel_data={"clientAddress": "203.0.113.7"})
        identity = extract_identity_from_activity(activity)
        assert identity.client_address == "203.0.113.7"

    def test_client_address_snake_case_key(self):
        activity = _agentic_activity(channel_data={"client_address": "203.0.113.8"})
        identity = extract_identity_from_activity(activity)
        assert identity.client_address == "203.0.113.8"

    def test_client_address_via_clientIp_fallback(self):
        activity = _agentic_activity(channel_data={"clientIp": "203.0.113.9"})
        identity = extract_identity_from_activity(activity)
        assert identity.client_address == "203.0.113.9"

    def test_client_address_via_sourceIp_fallback(self):
        activity = _agentic_activity(channel_data={"sourceIp": "203.0.113.10"})
        identity = extract_identity_from_activity(activity)
        assert identity.client_address == "203.0.113.10"

    def test_client_address_from_attr_shaped_channel_data(self):
        """``_lookup_mapping_or_attr`` walks both dict keys and object attributes."""
        channel_data = SimpleNamespace(clientAddress="203.0.113.11")
        activity = _agentic_activity(channel_data=channel_data)
        identity = extract_identity_from_activity(activity)
        assert identity.client_address == "203.0.113.11"

    def test_no_channel_data_yields_none(self):
        activity = _agentic_activity(channel_data=None)
        identity = extract_identity_from_activity(activity)
        assert identity.client_address is None


class TestTenantIdFallbackChain:
    """Tenant id is read from a chain of fallback locations."""

    def test_get_agentic_tenant_id_wins(self):
        activity = _agentic_activity(tenant="tenant-from-method")
        identity = extract_identity_from_activity(activity)
        assert identity.tenant_id == "tenant-from-method"

    def test_falls_back_to_recipient_tenant_id(self):
        activity = _agentic_activity(
            tenant=None,
            recipient_overrides={"tenant_id": "tenant-from-recipient"},
        )
        identity = extract_identity_from_activity(activity)
        assert identity.tenant_id == "tenant-from-recipient"

    def test_falls_back_to_conversation_tenant_id(self):
        conversation = SimpleNamespace(id="conv-1", tenant_id="tenant-from-conv")
        activity = _agentic_activity(tenant=None, conversation=conversation)
        identity = extract_identity_from_activity(activity)
        assert identity.tenant_id == "tenant-from-conv"

    def test_m18_falls_back_to_channel_data_tenant_id_dict(self):
        """M18 regression: Teams puts tenant id at ``channel_data.tenant.id``.

        Group-chat activities and older SDK versions leave the other fallbacks unset; this
        location is the canonical Bot Framework path and was previously not consulted.
        """
        activity = _agentic_activity(
            tenant=None,
            conversation=SimpleNamespace(id="conv-1", tenant_id=None),
            channel_data={"tenant": {
                "id": "tenant-from-channel-data"
            }},
        )
        identity = extract_identity_from_activity(activity)
        assert identity.tenant_id == "tenant-from-channel-data"

    def test_m18_falls_back_to_channel_data_tenant_id_attr_shaped(self):
        """The same fallback works when channel_data is object-shaped, not dict-shaped."""
        channel_data = SimpleNamespace(tenant=SimpleNamespace(id="tenant-attr-shape"))
        activity = _agentic_activity(
            tenant=None,
            conversation=SimpleNamespace(id="conv-1", tenant_id=None),
            channel_data=channel_data,
        )
        identity = extract_identity_from_activity(activity)
        assert identity.tenant_id == "tenant-attr-shape"

    def test_all_fallbacks_missing_yields_none_tenant(self):
        activity = _agentic_activity(
            tenant=None,
            conversation=SimpleNamespace(id="conv-1", tenant_id=None),
            channel_data=None,
        )
        identity = extract_identity_from_activity(activity)
        assert identity.tenant_id is None


class TestAgenticRoleCasing:
    """M17 regression: the recipient.role fallback must be case-insensitive."""

    def test_lowercase_agenticidentity_role_recognized(self):
        """JSON payloads from A365 sometimes lowercase the role name."""
        recipient = SimpleNamespace(
            role="agenticidentity",  # all lower
            agentic_app_id="agent-Y",
            agentic_user_id="user-Y",
            tenant_id="tenant-Y",
        )
        activity = SimpleNamespace(recipient=recipient, conversation=None)
        identity = extract_identity_from_activity(activity)
        assert identity is not None
        assert identity.agent_app_id == "agent-Y"

    def test_mixed_case_agenticuser_role_recognized(self):
        recipient = SimpleNamespace(
            role="AgenticUser",  # mixed
            agentic_app_id="agent-Z",
            agentic_user_id=None,
            tenant_id=None,
        )
        activity = SimpleNamespace(recipient=recipient, conversation=None)
        identity = extract_identity_from_activity(activity)
        assert identity is not None
        assert identity.agent_app_id == "agent-Z"

    def test_unrelated_role_still_rejected(self):
        recipient = SimpleNamespace(role="bot", agentic_app_id="x", tenant_id="y")
        activity = SimpleNamespace(recipient=recipient, conversation=None)
        assert extract_identity_from_activity(activity) is None

    def test_role_with_non_string_value_does_not_crash(self):
        """``str(role).lower()`` is defensive against unexpected types."""
        recipient = SimpleNamespace(role=42, agentic_app_id="x", tenant_id="y")
        activity = SimpleNamespace(recipient=recipient, conversation=None)
        assert extract_identity_from_activity(activity) is None


class TestCallableExceptionPaths:
    """m12 regression: warning logs include exception type so failure modes are greppable."""

    def test_is_agentic_request_raising_treated_as_non_agentic(self, caplog):
        """When ``is_agentic_request()`` raises, extraction returns None and warns with the type."""

        def boom():
            raise RuntimeError("simulated SDK breakage")

        activity = SimpleNamespace(
            is_agentic_request=boom,
            recipient=SimpleNamespace(role="agenticIdentity"),
        )

        with caplog.at_level(logging.WARNING, logger="nat.plugins.a365.turn_context"):
            result = extract_identity_from_activity(activity)

        assert result is None  # falls back to non-agentic, then bails
        matching = [r for r in caplog.records if "is_agentic_request() raised" in r.getMessage()]
        assert len(matching) == 1
        # m12 requirement: exception class name appears in the message so log queries find it.
        assert "RuntimeError" in matching[0].getMessage()

    def test_call_if_callable_raising_method_falls_back_to_none(self, caplog):
        """When ``get_agentic_instance_id()`` raises, we fall back to recipient.agentic_app_id."""

        def explode():
            raise ValueError("intentional")

        recipient = SimpleNamespace(
            role="agenticIdentity",
            agentic_app_id="fallback-agent",
            agentic_user_id=None,
            tenant_id="tenant-A",
        )
        activity = SimpleNamespace(
            is_agentic_request=lambda: True,
            get_agentic_instance_id=explode,
            get_agentic_tenant_id=lambda: None,
            get_agentic_user=lambda: None,
            recipient=recipient,
            conversation=None,
        )

        with caplog.at_level(logging.WARNING, logger="nat.plugins.a365.turn_context"):
            identity = extract_identity_from_activity(activity)

        # Recipient fallback supplied the agent id.
        assert identity is not None
        assert identity.agent_app_id == "fallback-agent"

        # m12: warning surfaces the exception type.
        matching = [r for r in caplog.records if "get_agentic_instance_id() raised" in r.getMessage()]
        assert len(matching) == 1
        assert "ValueError" in matching[0].getMessage()
