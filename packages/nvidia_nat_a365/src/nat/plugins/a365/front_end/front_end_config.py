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

"""Configuration for Microsoft Agent 365 front-end."""

import logging
from typing import Literal

from pydantic import Field, model_validator

from nat.data_models.common import OptionalSecretStr
from nat.data_models.front_end import FrontEndBaseConfig

logger = logging.getLogger(__name__)


class A365FrontEndConfig(FrontEndBaseConfig, name="a365"):
    """Microsoft Agent 365 front-end configuration.

    This front-end integrates NAT workflows with Microsoft Agent 365 hosting framework,
    enabling workflows to receive notifications from Teams, Email, and Office 365 apps.

    Authentication uses Entra ID (Azure AD) App Registration credentials (`app_id` and `app_password`)
    created when registering your bot in Azure Portal. The Microsoft Agents SDK authenticates with
    Entra ID via `MsalConnectionManager` to enable bot communication with Teams and Office 365.
    """

    host: str = Field(
        default="localhost",
        description="Host to bind the server to (default: localhost)"
    )
    port: int = Field(
        default=3978,
        description="Port to bind the server to (default: 3978)",
        ge=0,
        le=65535
    )
    app_id: str = Field(
        ...,
        description="Entra ID Application (client) ID from your Azure App Registration. "
                   "This is the Application ID created when registering your bot in Azure Portal."
    )
    app_password: OptionalSecretStr = Field(
        default=None,
        description="Entra ID client secret (password) from your Azure App Registration. "
                   "This authenticates your bot with Entra ID via Microsoft Bot Framework. "
                   "Can also be set via A365_APP_PASSWORD environment variable."
    )
    tenant_id: str | None = Field(
        default=None,
        description="Azure tenant ID (optional, defaults to 'common' for multi-tenant). "
                   "Specify your tenant ID for single-tenant apps, or leave None for multi-tenant."
    )
    log_level: str = Field(
        default="INFO",
        description="Log level for the server (default: INFO)"
    )
    enable_notifications: bool = Field(
        default=True,
        description="Enable A365 notification handlers (email, Word, Excel, PowerPoint, lifecycle)"
    )
    notification_workflow: str | None = Field(
        default=None,
        description="Optional workflow name to route notifications to. If not specified, uses the default workflow."
    )
    runner_class: str | None = Field(
        default=None,
        description="Custom worker class for handling A365 setup (default: built-in worker). "
                   "Specify as 'module.path.ClassName' to use a custom worker implementation."
    )

    @model_validator(mode="after")
    def validate_security_configuration(self):
        """Validate security configuration to prevent accidental misconfigurations."""
        localhost_hosts = {"localhost", "127.0.0.1", "::1"}
        
        # Warn if binding to non-localhost interface
        # Note: Microsoft Agents SDK handles authentication, but binding to public interfaces
        # should be done with caution and proper network security measures
        if self.host not in localhost_hosts:
            logger.warning(
                "A365 front-end is configured to bind to '%s' (non-localhost interface). "
                "Ensure proper network security measures are in place (firewall rules, "
                "reverse proxy with TLS, etc.). For local development, consider binding to localhost.",
                self.host
            )
        
        # Warn about default port in production-like scenarios
        if self.host not in localhost_hosts and self.port == 3978:
            logger.warning(
                "A365 front-end is using default port 3978 on a non-localhost interface. "
                "Consider using a non-standard port for production deployments."
            )
        
        return self
