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

from nat.cli.register_workflow import register_registry_handler
from nat.data_models.registry_handler import RegistryHandlerBaseConfig

logger = logging.getLogger(__name__)


class LocalRegistryHandlerConfig(RegistryHandlerBaseConfig, name="local"):
    """Interact with the local AIQ Toolkit environment to search and uninstall AIQ Toolkit components."""

    pass


@register_registry_handler(config_type=LocalRegistryHandlerConfig)
async def local_registry_handler(config: LocalRegistryHandlerConfig):

    from nat.registry_handlers.local.local_handler import LocalRegistryHandler

    registry_handler = LocalRegistryHandler()

    yield registry_handler
