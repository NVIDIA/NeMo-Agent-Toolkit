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

from aiq.builder.builder import Builder
from aiq.builder.function_info import FunctionInfo
from aiq.cli.register_workflow import register_function
from aiq.data_models.authentication import HTTPResponse
from aiq.data_models.component_ref import AuthenticationRef
from aiq.data_models.function import FunctionBaseConfig

logger = logging.getLogger(__name__)


class AuthenticatedRequestConfig(FunctionBaseConfig, name="authenticated_request"):
    """
    Simple function for making authenticated HTTP requests.

    Takes direct parameters for user_id, HTTP method, and URL to make authenticated requests or unauthenticated
    requests.
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
        Make HTTP requests using the AuthProviderMixin unified request method.

        This function demonstrates the AuthProviderMixin capabilities:
        - Unified request method for all HTTP methods
        - Authenticated requests as first citizen (default behavior)
        - Optional unauthenticated requests when needed
        - Returns actual API response content as text

        Args:
            url (str): The full URL to request
            method (str): HTTP method to use (GET, POST, PUT, DELETE, PATCH)
            user_id (str | None): User identifier for authentication.
            authenticated (bool): Whether to use authentication (defaults to True - first citizen)
            json_data (dict, optional): JSON data to send with POST/PUT/PATCH requests

        Returns:
            str: The actual API response content as text, or error message as text
        """
        try:
            # Use the unified request method from AuthProviderMixin
            response: HTTPResponse = await auth_provider.request(  # type: ignore[attr-defined]
                method=method,
                url=url,
                user_id=user_id if authenticated else None,
                apply_auth=authenticated,
                body_data=json_data if json_data else None)

            # Log the successful request
            auth_status = "authenticated" if authenticated else "public"
            logger.info(f"Successfully made {auth_status} {method.upper()} request to {url} "
                        f"(status: {response.status_code})")

            # Return only the actual API response content as text
            if response.body is not None:
                # If it's already a string, return it directly
                if isinstance(response.body, str):
                    return response.body

                # If it's a dict/object, return it as formatted JSON
                try:
                    return json.dumps(response.body, indent=2)
                except (TypeError, ValueError):
                    # If it can't be JSON serialized, convert to string
                    return str(response.body)
            else:
                return f"No response content received (HTTP {response.status_code})"

        except ValueError as e:
            # This catches user_id validation errors and other value errors
            error_msg = f"Validation error: {str(e)}"
            logger.error(error_msg)
            return error_msg

        except Exception as e:
            # Catch any other errors
            error_msg = f"Request failed: {str(e)}"
            logger.error(error_msg)
            return error_msg

    try:
        yield FunctionInfo.create(
            single_fn=_inner,
            description="Makes authenticated or unathenticated HTTP requests and returns the response content as text."
            "Authenticated requests are first citizen (default), with optional unauthenticated mode.")
    except GeneratorExit:
        logger.info("Authenticated request function exited early!")
    finally:
        logger.info("Cleaning up authenticated request function.")
