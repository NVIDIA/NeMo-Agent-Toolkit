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

from aiq.authentication.interfaces import AuthenticationClientBase
from aiq.builder.builder import Builder
from aiq.builder.function_info import FunctionInfo
from aiq.cli.register_workflow import register_function
from aiq.data_models.authentication import AuthResult
from aiq.data_models.function import FunctionBaseConfig

logger = logging.getLogger(__name__)


class HTTPAuthTool(FunctionBaseConfig, name="auth_tool"):
    """Authenticate to any registered API provider using OAuth2 authorization flow with browser consent handling."""
    pass


@register_function(config_type=HTTPAuthTool)
async def auth_tool(config: HTTPAuthTool, builder: Builder):
    """
    Uses HTTP Basic authentication to authenticate to any registered API provider.
    """

    async def _arun(authentication_provider_name: str) -> str:
        try:
            # Get the http basic auth registered authentication client
            basic_auth_client: AuthenticationClientBase = await builder.get_authentication(authentication_provider_name)

            # Perform authentication (this will invoke the user authentication callback)
            auth_context: AuthResult = await basic_auth_client.authenticate(user_id="default_user")

            if not auth_context or not auth_context.credentials:
                return f"Failed to authenticate provider: {authentication_provider_name}: Invalid credentials"

            return (f"Your registered API Provider name: [{authentication_provider_name}] is now authenticated.\n"
                    f"Credentials: {auth_context.model_dump()}.\n")

        except Exception as e:
            logger.exception("HTTP Basic authentication failed", exc_info=True)
            return f"HTTP Basic authentication to '{authentication_provider_name}' failed: {str(e)}"

    yield FunctionInfo.from_fn(_arun, description="Perform authentication with a given provider.")
