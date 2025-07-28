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

import httpx
from pydantic import ValidationError

from aiq.authentication.exceptions.call_back_exceptions import AuthenticationError
from aiq.authentication.exceptions.request_exceptions import BaseUrlValidationError
from aiq.authentication.exceptions.request_exceptions import BodyValidationError
from aiq.authentication.exceptions.request_exceptions import HTTPHeaderValidationError
from aiq.authentication.exceptions.request_exceptions import HTTPMethodValidationError
from aiq.authentication.exceptions.request_exceptions import QueryParameterValidationError
from aiq.authentication.interfaces import AuthProviderBaseConfig
from aiq.data_models.authentication import AuthResult
from aiq.data_models.authentication import HTTPMethod
from aiq.data_models.authentication import HTTPResponse
from aiq.utils.request_utils import HTTPRequestValidator

logger = logging.getLogger(__name__)


async def make_authenticated_request(url: str,
                                     auth_client: AuthProviderBaseConfig,
                                     user_id: str,
                                     method: str | HTTPMethod = HTTPMethod.GET,
                                     headers: str | dict | None = None,
                                     params: str | dict | None = None,
                                     body_data: str | dict | None = None,
                                     timeout: int | None = None) -> HTTPResponse:
    """
    Make an authenticated HTTP request to the specified URL.

    Extremely flexible function that accepts both string and dictionary inputs for maximum usability.

    Args:
        url (str): The URL to make the request to
        auth_client (AuthProviderBaseConfig): The authentication client to use for authentication
        user_id (Optional[str]): User ID for authentication. Uses 'default' if not specified.
        method (str | HTTPMethod): HTTP method (GET, POST, PUT, DELETE, etc.). Defaults to GET.
        headers (str | dict | None): Headers as JSON string or dictionary
        params (str | dict | None): Query parameters as JSON string or dictionary
        body_data (str | dict | None): Request body as JSON string or dictionary
        timeout (Optional[int]): Request timeout in seconds. Uses 30 if not specified.

    Returns:
        HTTPResponse: Structured response object containing response data, metadata, and authentication context
    """
    try:
        # Validate and request parameters
        HTTPRequestValidator.validate_base_url(url)
        validated_method = HTTPRequestValidator.validate_http_method(method)
        headers_dict = HTTPRequestValidator.validate_headers(headers)
        params_dict = HTTPRequestValidator.validate_query_parameters(params)
        json_data_dict = HTTPRequestValidator.validate_body_data(body_data)

        # Perform authentication using the auth provider
        auth_result: AuthResult = await auth_client.authenticate(user_id=user_id)

        if not auth_result or not auth_result.credentials:
            raise RuntimeError(f"Authentication failed for user '{user_id}': No credentials received")

        request_kwargs = {"headers": headers_dict or {}, "params": params_dict or {}, "timeout": timeout or 30}

        # Add JSON data if provided
        if json_data_dict is not None:
            request_kwargs["json"] = json_data_dict

        # Merge user-provided headers, params, cookies, and auth tuples with authentication credentials
        auth_result.attach(request_kwargs)

        # Make the authenticated request
        async with httpx.AsyncClient() as client:
            response = await client.request(validated_method, url, **request_kwargs)
            response.raise_for_status()

            # Parse response body and return unified result
            try:
                # First try to parse as JSON (could be dict, list, or other JSON types)
                response_body = response.json()
            except (json.JSONDecodeError, ValueError):
                # Fall back to text if JSON parsing fails
                response_body = response.text

            return HTTPResponse(status_code=response.status_code,
                                headers=dict(response.headers),
                                body=response_body,
                                cookies=dict(response.cookies) if response.cookies else None,
                                content_type=response.headers.get('Content-Type'),
                                url=str(response.url),
                                elapsed=response.elapsed.total_seconds() if response.elapsed else None,
                                auth_result=auth_result)

    except (BaseUrlValidationError,
            HTTPMethodValidationError,
            ValidationError,
            HTTPHeaderValidationError,
            QueryParameterValidationError,
            BodyValidationError) as e:

        error_msg = f"Request validation failed for {validated_method} {url}: {str(e)}"
        logger.error(error_msg)
        return HTTPResponse(
            status_code=400,  # Bad Request for validation errors
            headers={},
            body={
                "error": "Validation failed",
                "message": error_msg,
                "url": url,
                "method": validated_method,
                "status": "failed"
            },
            content_type="application/json",
            url=url)

    except AuthenticationError as e:
        error_msg = f"Authentication failed for {validated_method} {url}: {str(e)}"
        logger.error(error_msg)
        return HTTPResponse(
            status_code=401,  # Unauthorized for authentication errors
            headers={},
            body={
                "error": "Authentication failed",
                "message": error_msg,
                "url": url,
                "method": validated_method,
                "status": "failed"
            },
            content_type="application/json",
            url=url)

    except httpx.TimeoutException:
        error_msg = f"Request timeout while making {validated_method} request to {url}"
        logger.error(error_msg)
        return HTTPResponse(
            status_code=408,  # Request Timeout
            headers={},
            body={
                "error": "Request timeout",
                "message": error_msg,
                "url": url,
                "method": validated_method,
                "status": "failed"
            },
            content_type="application/json",
            url=url)

    except httpx.HTTPStatusError as e:
        error_msg = f"HTTP error {e.response.status_code} while making {validated_method} request to {url}"
        logger.error(error_msg)
        return HTTPResponse(
            status_code=e.response.status_code,  # Use actual HTTP status code
            headers=dict(e.response.headers) if e.response else {},
            body={
                "error": f"HTTP {e.response.status_code}",
                "message": error_msg,
                "url": url,
                "method": validated_method,
                "status": "failed"
            },
            content_type="application/json",
            url=url)

    except Exception as e:
        error_msg = f"Request failed for {validated_method} {url}: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return HTTPResponse(
            status_code=500,  # Internal Server Error for unexpected exceptions
            headers={},
            body={
                "error": str(e), "message": error_msg, "url": url, "method": validated_method, "status": "failed"
            },
            content_type="application/json",
            url=url)
