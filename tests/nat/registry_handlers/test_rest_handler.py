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

import base64
import json
import typing
from contextlib import AsyncExitStack
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest
from pytest_httpserver import HTTPServer

from nat.cli.type_registry import TypeRegistry
from nat.data_models.component import ComponentEnum
from nat.data_models.discovery_metadata import DiscoveryMetadata
from nat.registry_handlers.schemas.package import PackageNameVersionList
from nat.registry_handlers.schemas.publish import Artifact
from nat.registry_handlers.schemas.publish import BuiltArtifact
from nat.registry_handlers.schemas.pull import PullRequestPackages
from nat.registry_handlers.schemas.search import SearchQuery
from nat.settings.global_settings import Settings


@pytest.mark.parametrize("url, route, status, expected",
                         [
                             (None, "/publish", "success", "success"),
                             (None, "/publish", "error", "error"),
                             (None, "/publish", 1, "error"),
                             ("http://localhost:1234/badurl", "/publish", "bad_success", "error"),
                             (None, "/badroute", "success", "error"),
                         ])
@pytest.mark.usefixtures("httpserver_listen_address")
@pytest.mark.asyncio
async def test_rest_handler_publish(rest_registry_channel: dict,
                                    registry: TypeRegistry,
                                    global_settings: Settings,
                                    url: str | None,
                                    route: str,
                                    status: str,
                                    expected: str,
                                    httpserver: HTTPServer):

    publish_response_dict = {"status": {"status": status, "action": "publish", "message": ""}}
    publish_response_json = json.dumps(publish_response_dict)
    httpserver.expect_request(route).respond_with_data(publish_response_json)

    if url is None:
        url = httpserver.url_for(route)[:-len(route)]

    rest_registry_channel["channels"]["rest_channel"]["endpoint"] = url
    registry_config = global_settings.model_validate(rest_registry_channel)
    rest_registry_config = registry_config.channels.get("rest_channel", None)

    assert rest_registry_config is not None

    # Generate sample metadata
    metadata = {}
    for component_type in ComponentEnum:
        metadata[component_type] = []
        for i in range(3):
            metadata[component_type].append(
                DiscoveryMetadata(component_type=component_type, component_name=f"{component_type.value}_{i}"))

    built_artifact = BuiltArtifact(whl="base64encodedwhl", metadata=metadata)
    artifact = Artifact(artifact=built_artifact, whl_path="whl/path.whl")

    async with AsyncExitStack() as stack:
        registry_handler_info = registry.get_registry_handler(type(rest_registry_config))
        registry_handler = await stack.enter_async_context(registry_handler_info.build_fn(rest_registry_config))
        publish_response = await stack.enter_async_context(registry_handler.publish(artifact=artifact))

    assert publish_response.status.status == expected


@patch("nat.registry_handlers.rest.rest_handler.subprocess.run")
@pytest.mark.parametrize("url, route, return_value, expected",
                         [
                             (None, "/pull", 0, "success"),
                             (None, "/badroute", 0, "error"),
                             ("http://localhost:1234/badendpoint", "/pull", 0, "error"),
                         ])
@pytest.mark.usefixtures("httpserver_listen_address")
@pytest.mark.asyncio
async def test_rest_handler_pull(mock_run: MagicMock,
                                 rest_registry_channel: dict,
                                 registry: TypeRegistry,
                                 global_settings: Settings,
                                 url: str,
                                 route: str,
                                 expected: str,
                                 return_value: int,
                                 httpserver: HTTPServer):

    mock_stdout = MagicMock()
    mock_stdout.configure_mock(**{"method.return_value": return_value})

    sample_string = "Notional base64 string"
    sample_string_bytes = sample_string.encode("utf-8")

    base64_bytes = base64.b64encode(sample_string_bytes)
    base64_string = base64_bytes.decode("utf-8")

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

    pull_response_dict = {
        "packages": [{
            "whl": base64_string, "whl_name": "whl_name"
        }],
        "status": {
            "status": "success", "message": "", "action": "pull"
        }
    }
    pull_response_json = json.dumps(pull_response_dict)
    httpserver.expect_request(route).respond_with_data(pull_response_json)

    if (url is None):
        url = httpserver.url_for(route)[:-len(route)]

    rest_registry_channel["channels"]["rest_channel"]["endpoint"] = url
    registry_config = global_settings.model_validate(rest_registry_channel)
    rest_registry_config = registry_config.channels.get("rest_channel", None)

    pull_request_pkgs = PullRequestPackages(**pull_request_pkgs_dict)

    async with AsyncExitStack() as stack:
        registry_handler_info = registry.get_registry_handler(type(rest_registry_config))
        registry_handler = await stack.enter_async_context(registry_handler_info.build_fn(rest_registry_config))
        pull_response = await stack.enter_async_context(registry_handler.pull(packages=pull_request_pkgs))

    assert pull_response.status.status == expected


@pytest.mark.parametrize("url, route, field_name, component_type, top_k, expected",
                         [
                             (None, "/search", "all", "function", 10, "success"),
                             (None, "/search", "description", "function", 10, "success"),
                             (None, "/search", "component_name", "function", -1, "success"),
                             ("http://localhost:1234/badendpoint", "/search", "all", "function", 10, "error"),
                         ])
@pytest.mark.usefixtures("httpserver_listen_address")
@pytest.mark.asyncio
async def test_rest_handler_search(rest_registry_channel: dict,
                                   registry: TypeRegistry,
                                   global_settings: Settings,
                                   url: str | None,
                                   route: str,
                                   field_name: str,
                                   component_type: str,
                                   top_k: typing.Any,
                                   expected: str,
                                   httpserver: HTTPServer):

    route = "/search"
    search_query_dict = {"query": "*", "fields": [field_name], "component_types": [component_type], "top_k": top_k}

    search_response_dict = {
        "results": [{
            "package": "package_name",
            "version": "1.2.3",
            "component_type": "function",
            "component_name": "component_name",
            "description": "component description",
            "developer_notes": "developer notes"
        }],
        "params": {
            "query": "*", "fields": [field_name], "component_types": [component_type], "top_k": top_k
        },
        "status": {
            "status": "success", "message": "", "action": "search"
        }
    }

    search_response_dump = json.dumps(search_response_dict)
    httpserver.expect_request(route).respond_with_data(search_response_dump)

    if (url is None):
        url = httpserver.url_for(route)[:-len(route)]

    rest_registry_channel["channels"]["rest_channel"]["endpoint"] = url
    registry_config = global_settings.model_validate(rest_registry_channel)
    rest_registry_config = registry_config.channels.get("rest_channel", None)

    search_query = SearchQuery(**search_query_dict)

    async with AsyncExitStack() as stack:
        registry_handler_info = registry.get_registry_handler(type(rest_registry_config))
        registry_handler = await stack.enter_async_context(registry_handler_info.build_fn(rest_registry_config))
        search_response = await stack.enter_async_context(registry_handler.search(query=search_query))

    assert search_response.status.status == expected


@pytest.mark.parametrize("url, status, expected",
                         [
                             (None, "success", "success"),
                             (None, "error", "error"),
                             (None, 1, "error"),
                             (None, "bad_success", "error"),
                             ("http://localhost:1234/badendpoint", "success", "error"),
                         ])
@pytest.mark.usefixtures("httpserver_listen_address")
@pytest.mark.asyncio
async def test_rest_handler_remove(rest_registry_channel: dict,
                                   registry: TypeRegistry,
                                   global_settings: Settings,
                                   url: str | None,
                                   status: str,
                                   expected: str,
                                   httpserver: HTTPServer):

    route = "/remove"
    response_request_dict = {"packages": [{"name": "nat_package_name", "version": "1.2.3"}]}

    remove_response_dict = {
        "status": {
            "status": status, "message": "", "action": "remove"
        },
        "packages": [{
            "name": "nat_package_name", "version": "1.2.3"
        }]
    }

    search_response_dump = json.dumps(remove_response_dict)
    httpserver.expect_request(route).respond_with_data(search_response_dump)

    if (url is None):
        url = httpserver.url_for(route)[:-len(route)]

    rest_registry_channel["channels"]["rest_channel"]["endpoint"] = url
    registry_config = global_settings.model_validate(rest_registry_channel)
    rest_registry_config = registry_config.channels.get("rest_channel", None)

    package_name_version_list = PackageNameVersionList(**response_request_dict)

    async with AsyncExitStack() as stack:
        registry_handler_info = registry.get_registry_handler(type(rest_registry_config))
        registry_handler = await stack.enter_async_context(registry_handler_info.build_fn(rest_registry_config))
        remove_response = await stack.enter_async_context(registry_handler.remove(packages=package_name_version_list))

    assert remove_response.status.status == expected
