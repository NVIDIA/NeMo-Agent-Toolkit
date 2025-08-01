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
from pydantic import Field

from aiq.builder.builder import Builder
from aiq.builder.function_info import FunctionInfo
from aiq.cli.register_workflow import register_function
from aiq.data_models.authentication import BearerTokenCred
from aiq.data_models.component_ref import AuthenticationRef
from aiq.data_models.function import FunctionBaseConfig

logger = logging.getLogger(__name__)


class WhoAmIConfig(FunctionBaseConfig, name="who_am_i"):
    """
    Function that looks up the user's identity.
    """
    auth_provider: AuthenticationRef = Field(description=("Reference to the authentication provider to use for "
                                                          "authentication before making the who am i request."))

    api_url: str = Field(default="http://localhost:5001/api/me", description="Base URL for the who am i API")
    timeout: int = Field(default=10, description="Request timeout in seconds")


@register_function(config_type=WhoAmIConfig)
async def who_am_i_function(config: WhoAmIConfig, builder: Builder):

    auth_provider = await builder.get_auth_provider(config.auth_provider)

    async def _inner(empty: str = "") -> str:
        """
        Look up information about the currently logged in user.

        Returns:
            str: JSON string containing user information including name, email,
                 and other profile details from the OAuth provider
        """
        try:

            # Trigger the authentication flow
            auth_result = await auth_provider.authenticate()

            auth_header: BearerTokenCred = auth_result.credentials[0]

            async with httpx.AsyncClient(timeout=config.timeout) as client:
                response = await client.get(config.api_url,
                                            headers={"Authorization": f"Bearer {auth_header.token.get_secret_value()}"})
                response.raise_for_status()

                data = response.json()

                logger.info(f"Successfully looked up user: {data.get('name', 'Unknown')}")

                return json.dumps(data, indent=2)

        except httpx.TimeoutException:
            error_msg = "Request timeout while looking up user"
            logger.error(error_msg)
            return json.dumps({"error": "Request timeout", "status": "failed"})
        except httpx.HTTPStatusError as e:
            error_msg = f"HTTP error {e.response.status_code} while looking up user"
            logger.error(error_msg)
            return json.dumps({"error": f"HTTP {e.response.status_code}", "status": "failed"})
        except Exception as e:
            error_msg = f"Unexpected error looking up user: {str(e)}"
            logger.error(error_msg)
            return json.dumps({"error": str(e), "status": "failed"})

    try:
        yield FunctionInfo.create(single_fn=_inner, description="Look up who the currently logged in user is.")
    except GeneratorExit:
        logger.info("IP lookup function exited early!")
    finally:
        logger.info("Cleaning up IP lookup function.")
