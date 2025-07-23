# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from aiq.builder.builder import Builder
from aiq.builder.function_info import FunctionInfo
from aiq.cli.register_workflow import register_function
from aiq.data_models.function import FunctionBaseConfig


class RequestAttributesTool(FunctionBaseConfig, name="current_request_attributes"):
    """
    A simple tool that demonstrates how to retrieve user-defined request attributes from HTTP requests
    within workflow tools. Please refer to the 'general' section of the configuration file located in the
    'examples/getting_started/simple_web_query/configs/config-metadata.yml' directory to see how to define a
    custom route using a YAML file and associate it with a corresponding function to acquire request attributes.
    """
    pass


@register_function(config_type=RequestAttributesTool)
async def current_request_attributes(config: RequestAttributesTool, builder: Builder):

    from starlette.datastructures import Headers
    from starlette.datastructures import QueryParams

    async def _get_request_attributes(unused: str) -> str:

        from aiq.builder.context import AIQContext
        aiq_context = AIQContext.get()

        method: str | None = aiq_context.metadata.method
        url_path: str | None = aiq_context.metadata.url_path
        url_scheme: str | None = aiq_context.metadata.url_scheme
        headers: Headers | None = aiq_context.metadata.headers
        query_params: QueryParams | None = aiq_context.metadata.query_params
        path_params: dict[str, str] | None = aiq_context.metadata.path_params
        client_host: str | None = aiq_context.metadata.client_host
        client_port: int | None = aiq_context.metadata.client_port
        cookies: dict[str, str] | None = aiq_context.metadata.cookies
        conversation_id: str | None = aiq_context.conversation_id

        return (f"Method: {method}, "
                f"URL Path: {url_path}, "
                f"URL Scheme: {url_scheme}, "
                f"Headers: {dict(headers) if headers is not None else 'None'}, "
                f"Query Params: {dict(query_params) if query_params is not None else 'None'}, "
                f"Path Params: {path_params}, "
                f"Client Host: {client_host}, "
                f"Client Port: {client_port}, "
                f"Cookies: {cookies}, "
                f"Conversation Id: {conversation_id}")

    yield FunctionInfo.from_fn(_get_request_attributes,
                               description="Returns the acquired user defined request attriubutes.")
