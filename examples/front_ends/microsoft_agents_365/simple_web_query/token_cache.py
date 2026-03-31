# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import logging

logger = logging.getLogger(__name__)
_token_cache: dict[str, str] = {}


def cache_agentic_token(tenant_id: str, agent_id: str, token: str) -> None:
    _token_cache[f"{tenant_id}:{agent_id}"] = token


def get_cached_agentic_token(tenant_id: str, agent_id: str) -> str | None:
    return _token_cache.get(f"{tenant_id}:{agent_id}")


def clear_token_cache() -> None:
    _token_cache.clear()
