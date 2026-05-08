# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for A365 per-turn identity contextvar and activity extraction."""

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
