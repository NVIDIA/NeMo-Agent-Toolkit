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

import os

from pydantic import Field

from nat.cli.register_workflow import register_registry_handler
from nat.data_models.common import OptionalSecretStr
from nat.data_models.common import get_secret_value
from nat.data_models.registry_handler import RegistryHandlerBaseConfig


class RestRegistryHandlerConfig(RegistryHandlerBaseConfig, name="rest"):
    """Registry handler for interacting with a remote REST registry."""

    endpoint: str = Field(description="A string representing the remote endpoint.")
    token: OptionalSecretStr = Field(default=None,
                                     description="The authentication token to use when interacting with the registry.")
    publish_route: str = Field(default="", description="The route to the NAT publish service.")
    pull_route: str = Field(default="", description="The route to the NAT pull service.")
    search_route: str = Field(default="", description="The route to the NAT search service")
    remove_route: str = Field(default="", description="The route to the NAT remove service")


@register_registry_handler(config_type=RestRegistryHandlerConfig)
async def rest_search_handler(config: RestRegistryHandlerConfig):

    from nat.registry_handlers.rest.rest_handler import RestRegistryHandler

    if (config.token is None):
        registry_token = os.getenv("REGISTRY_TOKEN")

        if (registry_token is None):
            raise ValueError("Please supply registry token.")
    else:
        registry_token = get_secret_value(config.token)

    registry_handler = RestRegistryHandler(token=registry_token,
                                           endpoint=config.endpoint,
                                           publish_route=config.publish_route,
                                           pull_route=config.pull_route,
                                           search_route=config.search_route,
                                           remove_route=config.remove_route)

    yield registry_handler
