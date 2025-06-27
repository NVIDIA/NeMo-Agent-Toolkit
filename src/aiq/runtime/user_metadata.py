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

from starlette.datastructures import Headers
from starlette.datastructures import QueryParams

from aiq.data_models.api_server import Request


class RequestAttributes:
    """
    The RequestAttributes class is responsible for managing user-defined
    metadata and attributes. It provides a way to store and
    expose user-defined attributes to workflow tools.
    """

    def __init__(self) -> None:
        self._request: Request = Request()

    @property
    def method(self) -> str | None:
        """
        This property retrieves the HTTP method of the request.
        It can be GET, POST, PUT, DELETE, etc.

        Returns:
            str | None
        """
        return self._request.method

    @property
    def url_path(self) -> str | None:
        """
        This property retrieves the path from the URL of the request.

        Returns:
            str | None
        """
        return self._request.url_path

    @property
    def url_port(self) -> int | None:
        """
        This property retrieves the port number from the URL of the request.

        Returns:
            int | None
        """
        return self._request.url_port

    @property
    def url_scheme(self) -> str | None:
        """
        This property retrieves the scheme from the URL of the request.

        Returns:
            str | None
        """
        return self._request.url_scheme

    @property
    def headers(self) -> Headers | None:
        """
        This property retrieves the headers from the request stored in a dictionary-like object.

        Returns:
            Headers | None
        """
        return self._request.headers

    @property
    def query_params(self) -> QueryParams | None:
        """
        This property retrieves the query parameters from the request stored in a dictionary-like object.

        Returns:
            QueryParams | None
        """
        return self._request.query_params

    @property
    def path_params(self) -> dict[str, str] | None:
        """
        This property retrieves the path parameters from the request stored in a dictionary-like object.

        Returns:
            dict[str, str] | None
        """
        return self._request.path_params

    @property
    def client_host(self) -> str | None:
        """
        This property retrieves the clients remote hostname or IP address.

        Returns:
            str | None
        """
        return self._request.client_host

    @property
    def client_port(self) -> int | None:
        """
        This property retrieves the clients remote port number from which the client is connecting to.

        Returns:
            int | None
        """
        return self._request.client_port

    @property
    def cookies(self) -> dict[str, str] | None:
        """
        This property retrieves the cookies from the request stored in a dictionary-like object.

        Returns:
            dict[str, str] | None
        """
        return self._request.cookies
