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

from pydantic import Field

from aiq.authentication.interfaces import AuthProviderBase
from aiq.builder.builder import Builder
from aiq.builder.function_info import FunctionInfo
from aiq.cli.register_workflow import register_function
from aiq.data_models.authentication import AuthResult
from aiq.data_models.component_ref import AuthenticationRef
from aiq.data_models.function import FunctionBaseConfig

logger = logging.getLogger(__name__)


class AuthTool(FunctionBaseConfig, name="auth_tool"):
    """Authenticate to any registered API provider using OAuth2 authorization flow with browser consent handling."""
    auth_provider: AuthenticationRef = Field(description="Reference to the authentication provider "
                                             "to use for authentication.")


@register_function(config_type=AuthTool)
async def auth_tool(config: AuthTool, builder: Builder):
    """
    Uses HTTP Basic authentication to authenticate to any registered API provider.
    """
    basic_auth_client: AuthProviderBase = await builder.get_auth_provider(config.auth_provider)

    async def _arun(user_id: str) -> AuthResult:
        try:
            # Perform authentication (this will invoke the user authentication callback)
            auth_context: AuthResult = await basic_auth_client.authenticate(user_id=user_id)

            if not auth_context or not auth_context.credentials:
                raise RuntimeError(f"Failed to authenticate user: {user_id}: Invalid credentials")

            return auth_context

        except Exception as e:
            logger.exception("HTTP Basic authentication failed", exc_info=True)
            return f"HTTP Basic authentication for '{user_id}' failed: {str(e)}"

    yield FunctionInfo.from_fn(_arun, description="Perform authentication with a given user ID.")
