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

import logging
import json

import httpx
from pydantic import Field

from aiq.builder.builder import Builder
from aiq.builder.function_info import FunctionInfo
from aiq.cli.register_workflow import register_function
from aiq.data_models.component_ref import FunctionRef
from aiq.data_models.function import FunctionBaseConfig

logger = logging.getLogger(__name__)


class IPLookupFunctionConfig(FunctionBaseConfig, name="ip_lookup"):
    """
    Function that looks up information about an IP address using ip-api.com service.
    """
    authentication_function: FunctionRef = Field(description="Reference to the authentication function to perform "
                                                             "authentication before making the IP lookup request.")
    api_base_url: str = Field(
        default="http://ip-api.com/json",
        description="Base URL for the IP lookup API"
    )
    timeout: int = Field(
        default=10,
        description="Request timeout in seconds"
    )


@register_function(config_type=IPLookupFunctionConfig)
async def ip_lookup_function(
        config: IPLookupFunctionConfig, builder: Builder
):
    """
    Function that provides IP address lookup capabilities.
    """

    authentication_function = builder.get_function(config.authentication_function)

    async def _lookup_ip(ip_address: str) -> str:
        """
        Look up information about an IP address.

        Args:
            ip_address (str): The IP address to look up (e.g., "8.8.8.8")

        Returns:
            str: JSON string containing IP address information including country,
                 region, city, ISP, timezone, and other details
        """
        try:

            # Ensure the authentication function is called before making the request
            if authentication_function:
                logger.info(f"Calling authentication function")
                # Call the authentication function with a default user ID
                # We also don't store the credentials because we don't need them for this function
                _ = await authentication_function.acall_invoke(user_id="default")

            async with httpx.AsyncClient(timeout=config.timeout) as client:
                url = f"{config.api_base_url}/{ip_address}"
                logger.info(f"Looking up IP address: {ip_address} at {url}")

                response = await client.get(url)
                response.raise_for_status()

                data = response.json()

                # Check if the API returned an error
                if data.get("status") == "fail":
                    error_msg = data.get("message", "Unknown error from IP lookup service")
                    logger.error(f"IP lookup failed for {ip_address}: {error_msg}")
                    return json.dumps({
                        "error": error_msg,
                        "ip": ip_address,
                        "status": "failed"
                    })

                logger.info(f"Successfully looked up IP {ip_address}: {data.get('country', 'Unknown')}")
                return json.dumps(data, indent=2)

        except httpx.TimeoutException:
            error_msg = f"Request timeout while looking up IP {ip_address}"
            logger.error(error_msg)
            return json.dumps({
                "error": "Request timeout",
                "ip": ip_address,
                "status": "failed"
            })
        except httpx.HTTPStatusError as e:
            error_msg = f"HTTP error {e.response.status_code} while looking up IP {ip_address}"
            logger.error(error_msg)
            return json.dumps({
                "error": f"HTTP {e.response.status_code}",
                "ip": ip_address,
                "status": "failed"
            })
        except Exception as e:
            error_msg = f"Unexpected error looking up IP {ip_address}: {str(e)}"
            logger.error(error_msg)
            return json.dumps({
                "error": str(e),
                "ip": ip_address,
                "status": "failed"
            })

    try:
        yield FunctionInfo.create(single_fn=_lookup_ip, description="Look up information about an IP address.")
    except GeneratorExit:
        logger.info("IP lookup function exited early!")
    finally:
        logger.info("Cleaning up IP lookup function.")