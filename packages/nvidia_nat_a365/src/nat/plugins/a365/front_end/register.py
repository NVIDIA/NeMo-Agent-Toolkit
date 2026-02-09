# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
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

"""Registration for Microsoft Agent 365 front-end."""

from nat.cli.register_workflow import register_front_end
from nat.data_models.common import set_secret_from_env
from nat.plugins.a365.front_end.front_end_config import A365FrontEndConfig
from nat.plugins.a365.front_end.plugin import A365FrontEndPlugin


@register_front_end(config_type=A365FrontEndConfig)
async def a365_front_end(config: A365FrontEndConfig, full_config):
    """Register the Microsoft Agent 365 front-end.

    This front-end integrates NAT workflows with Microsoft Agent 365 hosting framework,
    enabling workflows to receive notifications from Teams, Email, and Office 365 apps.

    Args:
        config: A365FrontEndConfig with server and authentication settings
        full_config: Full NAT configuration

    Returns:
        A365FrontEndPlugin instance

    Raises:
        ValueError: If app_password is not provided in config or A365_APP_PASSWORD environment variable
    """
    # Load app_password from environment variable if not set in config
    if not config.app_password:
        set_secret_from_env(config, "app_password", "A365_APP_PASSWORD")
    
    # Validate that app_password is set
    if not config.app_password:
        raise ValueError(
            "app_password must be provided in the configuration or in the environment variable `A365_APP_PASSWORD`"
        )
    
    plugin = A365FrontEndPlugin(full_config=full_config)
    yield plugin
