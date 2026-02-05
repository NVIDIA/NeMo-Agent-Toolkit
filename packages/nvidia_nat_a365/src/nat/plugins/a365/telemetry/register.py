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

import importlib
import logging
from collections.abc import Callable
from typing import Optional

from pydantic import Field

from nat.builder.builder import Builder
from nat.cli.register_workflow import register_telemetry_exporter
from nat.data_models.telemetry_exporter import TelemetryExporterBaseConfig
from nat.observability.mixin.batch_config_mixin import BatchConfigMixin

logger = logging.getLogger(__name__)


def _resolve_token_resolver(token_resolver_path: Optional[str]) -> Optional[Callable[[str, str], Optional[str]]]:
    """Resolve a token resolver callable from a Python import path.

    Args:
        token_resolver_path: Python import path to callable (e.g., "module.path.function_name")
                           or None if no resolver provided

    Returns:
        Callable that takes (agent_id, tenant_id) and returns token string or None

    Raises:
        ValueError: If token_resolver_path is provided but invalid
        AttributeError: If function not found in module
    """
    if token_resolver_path is None:
        return None

    if not token_resolver_path.strip():
        raise ValueError("token_resolver path cannot be empty")

    # Split the function path to get module and function name
    try:
        module_path, function_name = token_resolver_path.rsplit(".", 1)
    except ValueError:
        raise ValueError(f"Invalid token_resolver path format: '{token_resolver_path}'. Expected 'module.path.function_name'")

    # Import the module
    try:
        module = importlib.import_module(module_path)
    except ImportError as e:
        raise ValueError(f"Failed to import module '{module_path}': {e}")

    # Get the function from the module
    if not hasattr(module, function_name):
        raise AttributeError(f"Function '{function_name}' not found in module '{module_path}'")

    token_resolver = getattr(module, function_name)

    if not callable(token_resolver):
        raise ValueError(f"'{token_resolver_path}' is not callable")

    return token_resolver


class A365TelemetryExporter(BatchConfigMixin, TelemetryExporterBaseConfig, name="a365"):
    """A telemetry exporter to transmit traces to Microsoft Agent 365 backend."""

    agent_id: str = Field(description="The Agent 365 agent ID")
    tenant_id: str = Field(description="The Azure tenant ID")
    token_resolver: Optional[str] = Field(
        default=None,
        description="Python callable path for token resolver function (agent_id, tenant_id) -> token"
    )
    cluster_category: str = Field(
        default="prod",
        description="Cluster category/environment (e.g., 'prod', 'dev')"
    )
    use_s2s_endpoint: bool = Field(
        default=False,
        description="Use service-to-service endpoint instead of standard endpoint"
    )
    suppress_invoke_agent_input: bool = Field(
        default=False,
        description="Suppress input messages for InvokeAgent spans"
    )


@register_telemetry_exporter(config_type=A365TelemetryExporter)
async def a365_telemetry_exporter(config: A365TelemetryExporter, builder: Builder):
    """Create an Agent 365 telemetry exporter.

    This is a stub implementation for initial plugin registration testing.
    Full implementation will integrate A365's _Agent365Exporter with NAT's telemetry system.
    """
    from nat.plugins.a365.telemetry.a365_exporter import A365OtelExporter

    # Resolve token resolver if provided
    token_resolver_callable = _resolve_token_resolver(config.token_resolver)

    if config.token_resolver and token_resolver_callable is None:
        logger.warning(
            f"Token resolver path '{config.token_resolver}' was provided but could not be resolved. "
            f"Telemetry export may fail without authentication."
        )

    logger.info(
        f"A365 telemetry exporter stub initialized for agent_id={config.agent_id}, "
        f"tenant_id={config.tenant_id}, cluster={config.cluster_category}, "
        f"token_resolver={'configured' if token_resolver_callable else 'none'}"
    )

    # Create stub exporter
    exporter = A365OtelExporter(
        agent_id=config.agent_id,
        tenant_id=config.tenant_id,
        token_resolver=token_resolver_callable,
        cluster_category=config.cluster_category,
        use_s2s_endpoint=config.use_s2s_endpoint,
        suppress_invoke_agent_input=config.suppress_invoke_agent_input,
        batch_size=config.batch_size,
        flush_interval=config.flush_interval,
        max_queue_size=config.max_queue_size,
        drop_on_overflow=config.drop_on_overflow,
        shutdown_timeout=config.shutdown_timeout,
    )

    yield exporter
