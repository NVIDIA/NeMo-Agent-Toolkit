# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from datetime import datetime
from datetime import timedelta
from datetime import timezone

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class A365AgenticTokenCacheEntry:
    token: str
    expires_at: datetime | None = None
    agentic_user_id: str | None = None


_CACHE_LOCK = threading.Lock()
_CACHE: dict[tuple[str, str], A365AgenticTokenCacheEntry] = {}


def _expires_at_utc(expires_at: datetime | None) -> datetime | None:
    if expires_at is None:
        return None
    if expires_at.tzinfo is not None:
        return expires_at.astimezone(timezone.utc)
    local_tz = datetime.now().astimezone().tzinfo
    return expires_at.replace(tzinfo=local_tz).astimezone(timezone.utc)


def cache_agentic_observability_token(
    tenant_id: str | None,
    agent_id: str | None,
    token: str | None,
    *,
    expires_at: datetime | None = None,
    agentic_user_id: str | None = None,
) -> bool:
    """Cache an Agent 365 agentic observability token for exporter callbacks."""
    if not tenant_id or not agent_id or not token:
        return False

    expires_utc = _expires_at_utc(expires_at)
    with _CACHE_LOCK:
        _CACHE[(tenant_id, agent_id)] = A365AgenticTokenCacheEntry(
            token=token,
            expires_at=expires_utc,
            agentic_user_id=agentic_user_id,
        )

    logger.info(
        "Cached agentic observability token (agent_id=%s, tenant_id=%s, agentic_user_id=%s, expires_at=%s)",
        agent_id,
        tenant_id,
        agentic_user_id,
        expires_utc.isoformat() if expires_utc else None,
    )
    return True


def get_cached_agentic_observability_token(
    tenant_id: str | None,
    agent_id: str | None,
    *,
    buffer_minutes: int = 5,
) -> str | None:
    """Return a valid cached agentic token for the tenant/agent pair, if present."""
    if not tenant_id or not agent_id:
        return None

    key = (tenant_id, agent_id)
    with _CACHE_LOCK:
        entry = _CACHE.get(key)
        if entry is None:
            return None

        if entry.expires_at is not None:
            buffer_time = datetime.now(timezone.utc) + timedelta(minutes=buffer_minutes)
            if entry.expires_at <= buffer_time:
                _CACHE.pop(key, None)
                logger.info(
                    "Cached agentic observability token expired or expiring soon "
                    "(agent_id=%s, tenant_id=%s, expires_at=%s)",
                    agent_id,
                    tenant_id,
                    entry.expires_at.isoformat(),
                )
                return None

        return entry.token


def clear_agentic_observability_token_cache() -> None:
    """Clear cached agentic observability tokens. Intended for tests."""
    with _CACHE_LOCK:
        _CACHE.clear()
