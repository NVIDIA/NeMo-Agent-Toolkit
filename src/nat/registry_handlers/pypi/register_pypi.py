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

from pydantic import Field

from nat.cli.register_workflow import register_registry_handler
from nat.data_models.registry_handler import RegistryHandlerBaseConfig


class PypiRegistryHandlerConfig(RegistryHandlerBaseConfig, name="pypi"):
    """Registry handler for interacting with a remote PyPI registry index."""

    endpoint: str = Field(description="A string representing the remote endpoint.")
    token: str | None = Field(default=None,
                              description="The authentication token to use when interacting with the registry.")
    publish_route: str = Field(description="The route to the NAT publish service.")
    pull_route: str = Field(description="The route to the NAT pull service.")
    search_route: str = Field(default="simple", description="The route to the NAT search service.")


@register_registry_handler(config_type=PypiRegistryHandlerConfig)
async def pypi_publish_registry_handler(config: PypiRegistryHandlerConfig):

    from nat.registry_handlers.pypi.pypi_handler import PypiRegistryHandler

    registry_handler = PypiRegistryHandler(endpoint=config.endpoint, token=config.token)

    yield registry_handler
