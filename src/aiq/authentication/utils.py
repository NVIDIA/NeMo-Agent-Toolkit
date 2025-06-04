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

import logging

import httpx

from aiq.authentication.exceptions import APIRequestError
from aiq.authentication.request_manager import RequestManager
from aiq.data_models.api_server import AuthenticatedRequest
from aiq.data_models.authentication import ExecutionMode

logger = logging.getLogger(__name__)


async def execute_api_request_default(request: AuthenticatedRequest) -> None:
    """
    Default callback handler for user requests. This no-op function raises a NotImplementedError error, to indicate that
    a valid request callback was not registered.

    Args:
        request (AuthenticatedRequest): The authenticated request to be executed.
    """
    raise NotImplementedError("No request callback was registered. Unable to handle request.")


async def execute_api_request_console(request: AuthenticatedRequest) -> httpx.Response | None:
    """
    Callback function that executes an API request in console mode using the provided authenticated request.

    Args:
        request (AuthenticatedRequest): The authenticated request to be executed.

    Returns:
        httpx.Response | None: The response from the API request, or None if an error occurs.
    """

    request_manager: RequestManager = RequestManager()
    response: httpx.Response | None = None

    request_manager.authentication_manager._set_execution_mode(ExecutionMode.CONSOLE)

    try:
        response = await request_manager._send_request(url=request.url_path,
                                                       http_method=request.method,
                                                       authentication_provider=request.authentication_provider,
                                                       headers=request.headers,
                                                       query_params=request.query_params,
                                                       body_data=request.body_data)

        if response is None:
            raise APIRequestError("An unexpected error occured while sending request.")

    except APIRequestError as e:
        logger.error("An error occured during the API request: %s", str(e), exc_info=True)
        return None

    return response


async def execute_api_request_server_http(request: AuthenticatedRequest) -> httpx.Response | None:
    """
    Callback function that executes an API request in http server mode using the provided authenticated request.

    Args:
        request (AuthenticatedRequest): The authenticated request to be executed.

    Returns:
        httpx.Response | None: The response from the API request, or None if an error occurs.
    """
    request_manager: RequestManager = RequestManager()
    response: httpx.Response | None = None

    request_manager.authentication_manager._set_execution_mode(ExecutionMode.SERVER)

    try:
        response = await request_manager._send_request(url=request.url_path,
                                                       http_method=request.method,
                                                       authentication_provider=request.authentication_provider,
                                                       headers=request.headers,
                                                       query_params=request.query_params,
                                                       body_data=request.body_data)

        if response is None:
            raise APIRequestError("An unexpected error occured while sending request.")

    except APIRequestError as e:
        logger.error("An error occured during the API request: %s", str(e), exc_info=True)
        return None

    return response
