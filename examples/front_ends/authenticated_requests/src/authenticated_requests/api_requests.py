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

import json
import logging
from typing import Literal

from pydantic import Field

from nat.builder.builder import Builder
from nat.builder.function_info import FunctionInfo
from nat.cli.register_workflow import register_function
from nat.data_models.authentication import HTTPResponse
from nat.data_models.component_ref import AuthenticationRef
from nat.data_models.function import FunctionBaseConfig

logger = logging.getLogger(__name__)


class AuthenticatedRequestConfig(FunctionBaseConfig, name="authenticated_request"):
    """
    Configuration for making authenticated or unauthenticated HTTP requests.
    """
    auth_provider: AuthenticationRef = Field(
        description="Reference to the authentication provider to use for API authentication.")


@register_function(config_type=AuthenticatedRequestConfig)
async def authenticated_request_function(config: AuthenticatedRequestConfig, builder: Builder):

    auth_provider = await builder.get_auth_provider(config.auth_provider)

    async def _inner(url: str,
                     method: Literal["GET", "POST", "PUT", "DELETE", "PATCH"] = "GET",
                     user_id: str | None = None,
                     authenticated: bool = True,
                     json_data: dict | None = None) -> str:
        """
        Make authenticated or public HTTP requests to external APIs.

        This function provides a unified interface for making HTTP requests with automatic
        authentication handling. It supports all standard HTTP methods and can switch
        between authenticated and public requests as needed.

        Args:
            url: The complete URL to send the request to (including protocol, domain, and path)
            method: HTTP method to use. Supported methods: GET, POST, PUT, DELETE, PATCH
            user_id: User identifier for authentication context.
            authenticated: Whether to apply authentication to this request. Defaults to True
            json_data: JSON payload to send with the request body. Typically used with
                      POST, PUT, or PATCH requests

        Returns:
            Response content as a formatted string.
        """
        try:
            response: HTTPResponse = await auth_provider.request(  # type: ignore[attr-defined]
                method=method,
                url=url,
                user_id=user_id if authenticated else None,
                apply_auth=authenticated,
                body_data=json_data if json_data else None)

            if response.body is not None:
                if isinstance(response.body, str):
                    formatted_response = response.body
                else:
                    try:
                        formatted_response = json.dumps(response.body, indent=2)
                    except (TypeError, ValueError):
                        formatted_response = str(response.body)

                if 200 <= response.status_code < 300:
                    return f"SUCCESS (HTTP {response.status_code}): {formatted_response}"
                elif 400 <= response.status_code < 500:
                    return f"CLIENT ERROR (HTTP {response.status_code}): {formatted_response}"
                elif 500 <= response.status_code < 600:
                    return f"SERVER ERROR (HTTP {response.status_code}): {formatted_response}"
                else:
                    return f"HTTP {response.status_code}: {formatted_response}"
            else:
                if response.status_code == 204:
                    return "SUCCESS (HTTP 204): No content - operation completed successfully"
                elif response.status_code == 202:
                    return "SUCCESS (HTTP 202): Request accepted for processing"
                else:
                    return f"No response content received (HTTP {response.status_code})"

        except ValueError as e:
            error_msg = f"VALIDATION ERROR: {str(e)}"
            logger.error("Validation error for %s %s: %s", method.upper(), url, str(e))
            return error_msg

        except Exception as e:
            error_msg = f"ERROR: {str(e)}"
            logger.error("Error for %s %s: %s", method.upper(), url, str(e))
            return error_msg

    yield FunctionInfo.create(
        single_fn=_inner,
        description=("Makes authenticated or unauthenticated HTTP requests and returns a formatted response string."
                     "Authenticated requests are the default behavior, with optional public/unauthenticated mode "
                     "for accessing public APIs."))
