# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from contextlib import AsyncExitStack
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest

from aiq.cli.type_registry import TypeRegistry
from aiq.registry_handlers.package_utils import build_aiq_artifact
from aiq.registry_handlers.pypi.pypi_handler import PypiRegistryHandler
from aiq.registry_handlers.schemas.pull import PullRequestPackages
from aiq.registry_handlers.schemas.search import SearchQuery
from aiq.settings.global_settings import Settings


@patch.object(PypiRegistryHandler, "_upload_to_pypi")
@pytest.mark.parametrize("return_value, expected", [
    (0, "success"),
    (1, "success"),
])
@pytest.mark.asyncio
async def test_pypi_handler_publish(mock_run: MagicMock,
                                    pypi_registry_channel: dict,
                                    registry: TypeRegistry,
                                    global_settings: Settings,
                                    return_value: int,
                                    expected: str):

    mock_stdout = MagicMock()
    mock_stdout.configure_mock(**{"method.return_value": return_value})

    package_root = "."

    registry_config = global_settings.model_validate(pypi_registry_channel)
    pypi_registry_config = registry_config.channels.get("pypi_channel", None)

    assert pypi_registry_config is not None

    artifact = build_aiq_artifact(package_root=package_root)

    async with AsyncExitStack() as stack:
        registry_handler_info = registry.get_registry_handler(type(pypi_registry_config))
        registry_handler = await stack.enter_async_context(registry_handler_info.build_fn(pypi_registry_config))
        publish_response = await stack.enter_async_context(registry_handler.publish(artifact=artifact))

    assert publish_response.status.status == expected


@patch("aiq.registry_handlers.pypi.pypi_handler.subprocess.run")
@pytest.mark.parametrize("return_value, expected", [
    (0, "success"),
    (1, "success"),
])
@pytest.mark.asyncio
async def test_pypi_handler_pull(mock_run: MagicMock,
                                 pypi_registry_channel: dict,
                                 registry: TypeRegistry,
                                 global_settings: Settings,
                                 return_value: int,
                                 expected: str):

    mock_stdout = MagicMock()
    mock_stdout.configure_mock(**{"method.return_value": return_value})

    pull_request_pkgs_dict = {
        "packages": [
            {
                "whl_path": "some_whl_path.whl"
            },
            {
                "name": "package_name", "version": "package_version"
            },
        ]
    }

    registry_config = global_settings.model_validate(pypi_registry_channel)
    pypi_registry_config = registry_config.channels.get("pypi_channel", None)

    assert pypi_registry_config is not None

    pull_request_pkgs = PullRequestPackages(**pull_request_pkgs_dict)

    async with AsyncExitStack() as stack:
        registry_handler_info = registry.get_registry_handler(type(pypi_registry_config))
        registry_handler = await stack.enter_async_context(registry_handler_info.build_fn(pypi_registry_config))
        pull_response = await stack.enter_async_context(registry_handler.pull(packages=pull_request_pkgs))

    assert pull_response.status.status == expected


@patch("aiq.registry_handlers.pypi.pypi_handler.subprocess.run")
@pytest.mark.parametrize("return_value, expected", [
    (0, "success"),
    (1, "success"),
])
@pytest.mark.asyncio
async def test_pypi_handler_search(mock_run: MagicMock,
                                   pypi_registry_channel: dict,
                                   registry: TypeRegistry,
                                   global_settings: Settings,
                                   return_value: int,
                                   expected: str):

    mock_stdout = MagicMock()
    mock_stdout.configure_mock(**{"method.return_value": return_value})

    search_query_dict = {"query": "*", "fields": ["all"], "component_types": ["function"], "top_k": -1}

    registry_config = global_settings.model_validate(pypi_registry_channel)
    pypi_registry_config = registry_config.channels.get("pypi_channel", None)

    assert pypi_registry_config is not None

    search_query = SearchQuery(**search_query_dict)

    async with AsyncExitStack() as stack:
        registry_handler_info = registry.get_registry_handler(type(pypi_registry_config))
        registry_handler = await stack.enter_async_context(registry_handler_info.build_fn(pypi_registry_config))
        search_response = await stack.enter_async_context(registry_handler.search(query=search_query))

    assert search_response.status.status == expected
