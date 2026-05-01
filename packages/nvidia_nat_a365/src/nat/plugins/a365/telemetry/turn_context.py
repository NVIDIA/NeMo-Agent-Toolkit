# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from contextvars import ContextVar
from contextvars import Token
from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class A365TurnTelemetryContext:
    """Per-turn Agent 365 telemetry identity and auth.

    This context is populated by the A365 front-end while a Teams/Agent 365 turn
    is being processed. The telemetry exporter reads it when available so a
    token resolved from TurnContext can override the static service token path.
    """

    agent_id: str | None = None
    tenant_id: str | None = None
    agentic_user_id: str | None = None
    token: str | None = None
    expires_at: datetime | None = None


_A365_TURN_TELEMETRY_CONTEXT: ContextVar[A365TurnTelemetryContext | None] = ContextVar(
    "a365_turn_telemetry_context",
    default=None,
)


def get_a365_turn_telemetry_context() -> A365TurnTelemetryContext | None:
    """Return the current per-turn Agent 365 telemetry context, if any."""
    return _A365_TURN_TELEMETRY_CONTEXT.get()


def set_a365_turn_telemetry_context(
    context: A365TurnTelemetryContext,
) -> Token[A365TurnTelemetryContext | None]:
    """Set the current per-turn Agent 365 telemetry context."""
    return _A365_TURN_TELEMETRY_CONTEXT.set(context)


def reset_a365_turn_telemetry_context(token: Token[A365TurnTelemetryContext | None]) -> None:
    """Reset the per-turn Agent 365 telemetry context."""
    _A365_TURN_TELEMETRY_CONTEXT.reset(token)
