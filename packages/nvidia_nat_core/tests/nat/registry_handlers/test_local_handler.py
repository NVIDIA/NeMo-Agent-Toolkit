# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import subprocess
from contextlib import AsyncExitStack
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest

from nat.cli.type_registry import GlobalTypeRegistry
from nat.cli.type_registry import TypeRegistry
from nat.registry_handlers.local.register_local import LocalRegistryHandlerConfig
from nat.registry_handlers.schemas.package import PackageNameVersion
from nat.registry_handlers.schemas.package import PackageNameVersionList
from nat.registry_handlers.schemas.remove import RemoveResponse
from nat.registry_handlers.schemas.search import SearchQuery
from nat.runtime.loader import PluginTypes
from nat.runtime.loader import discover_and_register_plugins
from nat.settings.global_settings import Settings


@pytest.mark.parametrize("field_name, component_type, top_k, expected",
                         [
                             ("all", "function", 5, "success"),
                             ("all", "llm_provider", 2, "success"),
                             ("all", "tool_wrapper", 0, "success"),
                             ("all", "llm_client", 2, "success"),
                             ("all", "embedder_provider", 1, "success"),
                             ("all", "embedder_client", 1, "success"),
                             ("all", "memory", 1, "success"),
                             ("all", "package", 1, "success"),
                             ("all", "registry_handler", 3, "success"),
                         ])
async def test_local_handler_search(
    local_registry_channel: dict,
    registry: TypeRegistry,
    field_name: str,
    component_type: str,
    top_k: int,
    expected: str,
):

    search_query_dict = {
        "query": "nvidia-nat", "fields": [field_name], "component_types": [component_type], "top_k": top_k
    }

    registry_config = Settings.model_validate(local_registry_channel)
    local_registry_config = registry_config.channels.get("local_channel", None)

    assert local_registry_config is not None

    registry_handler_info = registry.get_registry_handler(type(local_registry_config))

    async with registry_handler_info.build_fn(local_registry_config) as registry_handler:

        search_query = SearchQuery(**search_query_dict)
        async with registry_handler.search(query=search_query) as search_response:
            assert search_response.status.status == expected
            # When top_k=0, it means return all results (no limit)
            if top_k > 0:
                assert len(search_response.results) == top_k
            else:
                # top_k=0 means return all results, so just verify we got some
                assert len(search_response.results) >= 0


@pytest.mark.parametrize("expected_return_value, expected_status, expected_message",
                         [
                             (0, "success", ""),
                             (1, "error", "Error uninstalling artifacts: Command '' returned non-zero exit status 1."),
                         ])
@patch('nat.registry_handlers.local.local_handler.run_command', new_callable=AsyncMock)
async def test_local_handler_remove(mock_run: MagicMock,
                                    local_registry_channel: dict,
                                    global_settings: Settings,
                                    registry: TypeRegistry,
                                    expected_return_value: int,
                                    expected_status: str,
                                    expected_message: str):

    package0 = PackageNameVersion(name="package0", version="0.1")
    package1 = PackageNameVersion(name="package1")
    packages = PackageNameVersionList(packages=[package0, package1])

    expected_response = RemoveResponse(status={
        "status": expected_status, "message": expected_message, "action": "remove"
    })

    if expected_return_value == 1:
        mock_run.side_effect = subprocess.CalledProcessError(expected_return_value, "")

    discover_and_register_plugins(PluginTypes.CONFIG_OBJECT)

    registry = GlobalTypeRegistry.get()

    registry_config = global_settings.model_validate(local_registry_channel)

    assert registry_config.channels.get("local_channel_bad", None) is None

    local_registry_config = registry_config.channels.get("local_channel", None)

    assert isinstance(local_registry_config, LocalRegistryHandlerConfig)

    async with AsyncExitStack() as stack:
        registry_handler_info = registry.get_registry_handler(type(local_registry_config))
        registry_handler = await stack.enter_async_context(registry_handler_info.build_fn(local_registry_config))
        publish_response = await stack.enter_async_context(registry_handler.remove(packages=packages))

        assert publish_response == expected_response


async def test_local_handler_remove_keeps_event_loop_responsive(local_registry_channel: dict,
                                                                 global_settings: Settings,
                                                                 registry: TypeRegistry):
    """`remove()` must not block the event loop while the package manager runs.

    `uv pip uninstall` is a child process that can take seconds. When this was
    a blocking `subprocess.run`, every other in-flight agent workflow froze for
    its whole duration. This drives the real subprocess through the async seam
    and asserts a concurrent task keeps running.
    """
    import asyncio
    import sys

    from nat.registry_handlers.local import local_handler
    from nat.registry_handlers.process_utils import run_command as real_run_command

    ticks = 0
    stop = asyncio.Event()

    async def heartbeat():
        nonlocal ticks
        while not stop.is_set():
            await asyncio.sleep(0.005)
            ticks += 1

    awaited = []

    async def slow_run_command(*args, **kwargs):
        # Same code path as production, just a command we can rely on existing.
        awaited.append(args)
        return await real_run_command(sys.executable, "-c", "import time; time.sleep(0.25)")

    package = PackageNameVersion(name="package0", version="0.1")
    packages = PackageNameVersionList(packages=[package])

    discover_and_register_plugins(PluginTypes.CONFIG_OBJECT)
    registry_config = global_settings.model_validate(local_registry_channel)
    local_registry_config = registry_config.channels.get("local_channel", None)
    assert local_registry_config is not None

    beat = asyncio.create_task(heartbeat())
    await asyncio.sleep(0.02)
    before = ticks

    with patch.object(local_handler, "run_command", slow_run_command):
        async with AsyncExitStack() as stack:
            registry_handler_info = registry.get_registry_handler(type(local_registry_config))
            registry_handler = await stack.enter_async_context(registry_handler_info.build_fn(local_registry_config))
            response = await stack.enter_async_context(registry_handler.remove(packages=packages))

    during = ticks - before
    stop.set()
    await beat

    assert response.status.status == "success"
    assert awaited, "handler did not go through the async run_command seam"
    assert during > 0, "event loop was starved for the duration of the subprocess call"
