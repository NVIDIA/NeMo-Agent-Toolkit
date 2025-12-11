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
"""Registration of A2A front end with NAT plugin system."""

from collections.abc import AsyncIterator

from nat.cli.register_workflow import register_front_end
from nat.data_models.config import Config
from nat.plugins.a2a.server.front_end_config import A2AFrontEndConfig


@register_front_end(config_type=A2AFrontEndConfig)
async def register_a2a_front_end(_config: A2AFrontEndConfig, full_config: Config) -> AsyncIterator:
    """Register the A2A front end plugin.

    Args:
        _config: The A2A front end configuration (unused, provided for registration)
        full_config: The complete NAT configuration

    Yields:
        A2AFrontEndPlugin instance
    """
    from nat.plugins.a2a.server.front_end_plugin import A2AFrontEndPlugin

    yield A2AFrontEndPlugin(full_config=full_config)
