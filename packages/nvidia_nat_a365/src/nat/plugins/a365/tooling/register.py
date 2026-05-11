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
"""Registration for A365 tooling integration with NAT MCP client."""

from __future__ import annotations

import logging
from collections.abc import Awaitable
from collections.abc import Callable
from collections.abc import Sequence
from contextlib import AsyncExitStack
from typing import NoReturn
from urllib.parse import urlparse

from nat.builder.builder import Builder
from nat.builder.function import Function
from nat.builder.function import FunctionGroup
from nat.cli.register_workflow import register_function_group
from nat.data_models.component_ref import AuthenticationRef
from nat.plugins.a365.exceptions import A365AuthenticationError
from nat.plugins.a365.exceptions import A365ConfigurationError
from nat.plugins.a365.exceptions import A365SDKError
from nat.plugins.a365.tooling.tooling_config import A365MCPToolingConfig
from nat.plugins.a365.tooling.tooling_config import ServerRegistrationErrorPolicy

logger = logging.getLogger(__name__)

# Status codes that indicate the gateway considered our caller unauthenticated /
# unauthorized -- distinct from server-side or network failures that should be
# classified as SDK errors instead.
_AUTH_HTTP_STATUSES = frozenset({401, 403})


def _server_display_name(server: object) -> str:
    """Return a deterministic, human-readable name for an MCP server.

    Falls back to ``unknown:<hostname>`` when the gateway response omits
    ``mcp_server_name`` so two unnamed servers can be told apart in logs and
    ``server_auth_providers`` lookups. Prevents the "two servers, both called
    'unknown', both eligible for the same override" footgun.
    """
    name = getattr(server, "mcp_server_name", None)
    if name:
        return str(name)
    url = getattr(server, "url", None)
    if url:
        try:
            host = urlparse(str(url)).hostname
        except ValueError:
            host = None
        if host:
            return f"unknown:{host}"
    return "unknown"


def _raise_classified_discovery_error(error: Exception) -> NoReturn:
    """Classify a discovery error as auth vs. SDK and raise the matching A365 exception.

    Preference order:
    1. ``aiohttp.ClientResponseError`` with status 401/403 -> ``A365AuthenticationError``.
    2. Substring match on the error message ("authentication" / "unauthorized" /
       "forbidden") -> ``A365AuthenticationError`` (covers SDK wrappers that hide the
       underlying HTTP status from us).
    3. Everything else -> ``A365SDKError`` with a WARN that lets ops detect when the
       substring matcher is missing real failure types in production.
    """
    # 1. Strongly-typed HTTP failure from aiohttp.
    try:
        from aiohttp import ClientResponseError
    except ImportError:  # pragma: no cover -- aiohttp is a hard transitive dep
        ClientResponseError = ()  # type: ignore[assignment]
    if ClientResponseError and isinstance(error, ClientResponseError):
        status = getattr(error, "status", None)
        if status in _AUTH_HTTP_STATUSES:
            raise A365AuthenticationError(
                f"Failed to authenticate with A365 tooling gateway (HTTP {status}): {error}",
                original_error=error,
            ) from error
        raise A365SDKError(
            f"Failed to discover MCP servers from A365 tooling gateway (HTTP {status}): {error}",
            sdk_component="McpToolServerConfigurationService",
            original_error=error,
        ) from error

    # 2. Fallback: substring match on error message. Kept for compatibility with SDK wrappers.
    error_msg = str(error).lower()
    if any(token in error_msg for token in ("authentication", "unauthorized", "forbidden")):
        raise A365AuthenticationError(
            f"Failed to authenticate with A365 tooling gateway: {error}",
            original_error=error,
        ) from error

    # 3. Unclassified failure. Emit the exception type at WARN so a pattern of misses is
    # observable (e.g. a new SDK wrapper type that auth/network errors are now coming in as).
    logger.warning(
        "A365 tooling discovery raised an unclassified exception type %r; "
        "classifying as A365SDKError. If this is recurring, update the classifier.",
        type(error).__name__,
    )
    raise A365SDKError(
        f"Failed to discover MCP servers from A365 tooling gateway: {error}",
        sdk_component="McpToolServerConfigurationService",
        original_error=error,
    ) from error


def _handle_server_registration_error(
    *,
    server_name: str,
    error: Exception,
    policy: ServerRegistrationErrorPolicy,
    skipped_servers: list[tuple[str, str]],
) -> None:
    """Apply the configured ``on_server_registration_error`` policy.

    ``fail_fast`` raises ``A365SDKError`` so the caller's ``async with`` aborts.
    The two tolerant policies record ``(server_name, error_repr)`` into
    ``skipped_servers`` so the composite group exposes which servers were lost.
    """
    if policy == "fail_fast":
        raise A365SDKError(
            f"Failed to register MCP server '{server_name}' (policy=fail_fast): {error}",
            sdk_component="mcp_client_function_group",
            original_error=error,
        ) from error

    skipped_servers.append((server_name, repr(error)))

    if policy == "skip_with_warning":
        logger.warning(
            "Skipping MCP server %r after registration failure (policy=skip_with_warning): %s",
            server_name,
            error,
            exc_info=True,
        )
    else:  # skip_silently
        logger.debug(
            "Skipping MCP server %r after registration failure (policy=skip_silently): %s",
            server_name,
            error,
            exc_info=True,
        )


class A365MCPToolingFunctionGroup(FunctionGroup):
    """
    Composite FunctionGroup that aggregates functions from multiple MCP servers.

    Instead of merging functions into a single group, this class delegates to
    multiple MCP FunctionGroups and aggregates their results. This preserves
    the original function bindings and avoids double-wrapping.

    Skipped-server metadata: ``skipped_servers`` records ``(name, error_repr)``
    tuples for any servers that failed registration under the
    ``skip_with_warning`` / ``skip_silently`` policies. Use it for monitoring or
    health-check surfaces (`/health` endpoints, dashboards) so quiet degradation
    is observable.
    """

    def __init__(
        self,
        config: A365MCPToolingConfig,
        mcp_groups: list[FunctionGroup],
        skipped_servers: list[tuple[str, str]] | None = None,
    ):
        """
        Initialize the composite function group.

        Args:
            config: The A365 MCP tooling configuration
            mcp_groups: List of MCP FunctionGroups to aggregate
            skipped_servers: List of ``(server_name, error_repr)`` for servers that
                failed registration under a tolerant policy. Empty under ``fail_fast``.
        """
        super().__init__(config=config, instance_name="a365_mcp_tooling")
        self._mcp_groups = mcp_groups
        self.skipped_servers: list[tuple[str, str]] = list(skipped_servers or [])

    @staticmethod
    async def _aggregate(
        mcp_groups: list[FunctionGroup],
        method_name: str,
        filter_fn: Callable[[Sequence[str]], Awaitable[Sequence[str]]] | None,
    ) -> dict[str, Function]:
        """Merge ``method_name`` outputs across MCP groups, warning on name collisions.

        Function names from MCP groups are already namespaced (e.g.
        ``mcp_client__tool_name``), so collisions are unusual -- they typically
        indicate the same ``tool_overrides.alias`` was applied to two servers that
        each expose the original tool name. When that happens, the later definition
        wins (preserving prior dict-update behavior) but a WARN surfaces the
        ambiguity so the operator can disambiguate via per-server overrides.
        """
        merged: dict[str, Function] = {}
        for mcp_group in mcp_groups:
            method = getattr(mcp_group, method_name)
            group_functions = await method(filter_fn=filter_fn)
            for name, fn in group_functions.items():
                if name in merged and merged[name] is not fn:
                    logger.warning(
                        "Tool name collision in A365MCPToolingFunctionGroup.%s: %r appears in "
                        "multiple MCP groups; later definition wins. Disambiguate via "
                        "``tool_overrides`` or per-server configuration.",
                        method_name,
                        name,
                    )
                merged[name] = fn
        return merged

    async def get_all_functions(
        self,
        filter_fn: Callable[[Sequence[str]], Awaitable[Sequence[str]]] | None = None,
    ) -> dict[str, Function]:
        """Aggregate all functions from all MCP groups (collision-aware)."""
        return await self._aggregate(self._mcp_groups, "get_all_functions", filter_fn)

    async def get_accessible_functions(
        self,
        filter_fn: Callable[[Sequence[str]], Awaitable[Sequence[str]]] | None = None,
    ) -> dict[str, Function]:
        """Aggregate accessible functions from all MCP groups (collision-aware)."""
        return await self._aggregate(self._mcp_groups, "get_accessible_functions", filter_fn)

    async def get_included_functions(
        self,
        filter_fn: Callable[[Sequence[str]], Awaitable[Sequence[str]]] | None = None,
    ) -> dict[str, Function]:
        """Aggregate included functions from all MCP groups (collision-aware)."""
        return await self._aggregate(self._mcp_groups, "get_included_functions", filter_fn)

    async def get_excluded_functions(
        self,
        filter_fn: Callable[[Sequence[str]], Awaitable[Sequence[str]]] | None = None,
    ) -> dict[str, Function]:
        """Aggregate excluded functions from all MCP groups (collision-aware)."""
        return await self._aggregate(self._mcp_groups, "get_excluded_functions", filter_fn)


@register_function_group(config_type=A365MCPToolingConfig)
async def a365_mcp_tooling_function_group(config: A365MCPToolingConfig, builder: Builder):
    """Register MCP servers discovered from A365 tooling as NAT function groups.

    This function:
    1. Uses A365 tooling service to discover configured MCP servers
    2. Creates MCP client function groups for each discovered server
    3. Returns a composite function group containing all discovered tools

    Args:
        config: A365MCPToolingConfig with agent ID and auth token
        builder: NAT Builder instance

    Returns:
        FunctionGroup containing tools from all discovered MCP servers

    Raises:
        OptionalImportError: If nvidia-nat-mcp package is not installed
        A365AuthenticationError: If authentication fails when resolving auth token or discovering servers
        A365SDKError: If MCP server discovery fails
    """
    try:
        from nat.plugins.mcp.client.client_config import MCPClientConfig
        from nat.plugins.mcp.client.client_config import MCPServerConfig
    except ImportError as e:
        from nat.utils.optional_imports import OptionalImportError

        raise OptionalImportError(
            "nvidia-nat-mcp",
            additional_message=("The A365 tooling feature requires the MCP client functionality. "
                                "Install it with one of the following:\n"
                                "  - uv pip install nvidia-nat-mcp\n"
                                "  - uv pip install 'nvidia-nat[mcp]'\n"
                                "  - uv pip install 'nvidia-nat-a365[mcp]' (if installing from source)"),
        ) from e

    from nat.plugins.a365.tooling import A365ToolingService

    auth_token_str: str | None
    if isinstance(config.auth_token, AuthenticationRef):
        auth_provider = await builder.get_auth_provider(config.auth_token)

        # Get user_id from context if available (needed for OAuth flows).
        # ``Context.get().user_id`` is ``None`` outside an active turn, which is the normal
        # state for S2S / startup-time tooling discovery. Auth providers that require a
        # user_id (e.g. OBO/delegated flows) will surface that as their own exception.
        from nat.builder.context import Context
        user_id = Context.get().user_id

        auth_result = await auth_provider.authenticate(user_id=user_id)
        if not auth_result.credentials:
            raise A365AuthenticationError("No credentials available from auth provider")

        # Reuse the telemetry-side extractor so the (BearerTokenCred / HeaderCred)
        # contract stays in one place. If a future NAT credential type is added, only
        # one site needs updating.
        from nat.plugins.a365.telemetry.register import _default_token_extractor

        auth_token_str = _default_token_extractor(auth_result)

        if auth_token_str is None:
            raise A365AuthenticationError(
                f"No bearer token found in auth provider credentials. "
                f"Found credential types: {[type(c).__name__ for c in auth_result.credentials]}")
    else:
        auth_token_str = config.auth_token

    service = A365ToolingService()
    logger.info(f"Discovering MCP servers for agent {config.agentic_app_id}")
    try:
        servers = await service.list_tool_servers(
            agentic_app_id=config.agentic_app_id,
            auth_token=auth_token_str,
        )
    except Exception as e:
        _raise_classified_discovery_error(e)

    logger.info(f"Discovered {len(servers)} MCP servers, registering as function groups")

    from nat.plugins.mcp.client.client_impl import mcp_client_function_group

    # Convert tool_overrides dict to MCPToolOverrideConfig if provided (once, before the loop).
    # NOTE: the override map is applied identically to every server. Two servers exposing the
    # same source tool will both get the same alias, which can collide in the composite group.
    # ``A365MCPToolingFunctionGroup`` warns on such collisions at aggregation time.
    mcp_tool_overrides = None
    if config.tool_overrides:
        from pydantic import ValidationError

        from nat.plugins.mcp.client.client_config import MCPToolOverrideConfig

        try:
            mcp_tool_overrides = {
                tool_name: MCPToolOverrideConfig(**override)
                for tool_name, override in config.tool_overrides.items()
            }
        except (ValidationError, TypeError) as e:
            raise A365ConfigurationError(f"Invalid tool_overrides configuration: {str(e)}") from e

    # Normalize ``server_auth_providers`` keys to lower-case for case-insensitive lookup against
    # discovered server names. Track which override keys we actually use so unused entries can be
    # surfaced as misconfiguration after discovery.
    configured_overrides: dict[str, str | AuthenticationRef] = {
        k.lower(): v
        for k, v in (config.server_auth_providers or {}).items()
    }
    used_override_keys: set[str] = set()

    mcp_groups: list[FunctionGroup] = []
    skipped_servers: list[tuple[str, str]] = []

    # Use AsyncExitStack to keep all MCP client contexts open for the lifetime of this function group
    async with AsyncExitStack() as exit_stack:
        for server in servers:
            server_name = _server_display_name(server)

            if not server.url:
                logger.warning("Skipping server %s: no URL configured", server_name)
                continue

            # Priority: 1) Per-server override (case-insensitive), 2) A365 gateway auth (if
            # AuthenticationRef), 3) None.
            server_auth_provider = None
            override_key = server_name.lower()
            if override_key in configured_overrides:
                server_auth_provider = configured_overrides[override_key]
                used_override_keys.add(override_key)
                logger.debug("Using per-server auth provider %r for server %r", server_auth_provider, server_name)
            elif isinstance(config.auth_token, AuthenticationRef):
                server_auth_provider = config.auth_token
                logger.debug("Using A365 gateway auth provider for server %r", server_name)

            mcp_config = MCPClientConfig(
                server=MCPServerConfig(
                    transport="streamable-http",
                    url=server.url,
                    auth_provider=server_auth_provider,
                ),
                tool_call_timeout=config.tool_call_timeout,
                auth_flow_timeout=config.auth_flow_timeout,
                reconnect_enabled=config.reconnect_enabled,
                reconnect_max_attempts=config.reconnect_max_attempts,
                reconnect_initial_backoff=config.reconnect_initial_backoff,
                reconnect_max_backoff=config.reconnect_max_backoff,
                session_aware_tools=config.session_aware_tools,
                max_sessions=config.max_sessions,
                session_idle_timeout=config.session_idle_timeout,
                tool_overrides=mcp_tool_overrides,
            )

            # mcp_client_function_group is an async context manager; AsyncExitStack keeps
            # all contexts open for the lifetime of the composite group.
            try:
                mcp_group = await exit_stack.enter_async_context(mcp_client_function_group(mcp_config, builder))
                mcp_groups.append(mcp_group)
                logger.info("Registered MCP server %r", server_name)
            except Exception as e:
                _handle_server_registration_error(
                    server_name=server_name,
                    error=e,
                    policy=config.on_server_registration_error,
                    skipped_servers=skipped_servers,
                )

        # Warn about server_auth_providers keys that didn't match any discovered server.
        # Common cause: server was renamed in the gateway but the YAML wasn't updated.
        unused_override_keys = set(configured_overrides) - used_override_keys
        if unused_override_keys:
            logger.warning(
                "``server_auth_providers`` references unknown MCP servers (no override applied): %s. "
                "Update or remove these entries.",
                sorted(unused_override_keys),
            )

        if not mcp_groups:
            logger.warning(
                "No MCP servers successfully registered for agent %s. "
                "Discovered %d servers but none could be registered.",
                config.agentic_app_id,
                len(servers))

        composite_group = A365MCPToolingFunctionGroup(
            config=config,
            mcp_groups=mcp_groups,
            skipped_servers=skipped_servers,
        )

        all_functions = await composite_group.get_all_functions()
        logger.info(
            "A365 MCP tooling: registered %d total tools from %d servers (skipped=%d)",
            len(all_functions),
            len(mcp_groups),
            len(skipped_servers),
        )

        yield composite_group
