# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#
# OAuth callback / auth helper endpoints.

from __future__ import annotations

from typing import TYPE_CHECKING

import httpx
from authlib.common.errors import AuthlibBaseError as OAuthError
from fastapi import FastAPI
from fastapi import Request
from fastapi.responses import HTMLResponse

from nat.front_ends.fastapi.html_snippets.auth_code_grant_success import AUTH_REDIRECT_SUCCESS_HTML

if TYPE_CHECKING:
    from nat.front_ends.fastapi.fastapi_front_end_plugin_worker import FastApiFrontEndPluginWorker


async def register_auth_routes(worker: FastApiFrontEndPluginWorker, app: FastAPI) -> None:

    async def redirect_uri(request: Request):
        """
        Handle the redirect URI for OAuth2 authentication.
        Args:
            request: The FastAPI request object containing query parameters.

        Returns:
            HTMLResponse: A response indicating the success of the authentication flow.
        """
        state = request.query_params.get("state")

        async with worker._outstanding_flows_lock:
            if not state or state not in worker._outstanding_flows:
                return "Invalid state. Please restart the authentication process."

            flow_state = worker._outstanding_flows[state]

        config = flow_state.config
        verifier = flow_state.verifier
        client = flow_state.client

        try:
            res = await client.fetch_token(url=config.token_url,
                                           authorization_response=str(request.url),
                                           code_verifier=verifier,
                                           state=state)
            flow_state.future.set_result(res)
        except OAuthError as e:
            flow_state.future.set_exception(
                RuntimeError(f"Authorization server rejected request: {e.error} ({e.description})"))
        except httpx.HTTPError as e:
            flow_state.future.set_exception(RuntimeError(f"Network error during token fetch: {e}"))
        except Exception as e:
            flow_state.future.set_exception(RuntimeError(f"Authentication failed: {e}"))

        return HTMLResponse(content=AUTH_REDIRECT_SUCCESS_HTML,
                            status_code=200,
                            headers={
                                "Content-Type": "text/html; charset=utf-8", "Cache-Control": "no-cache"
                            })

    if (worker.front_end_config.oauth2_callback_path):
        # Add the redirect URI route
        worker._register_api_route(
            app,
            path=worker.front_end_config.oauth2_callback_path,
            endpoint=redirect_uri,
            methods=["GET"],
            description="Handles the authorization code and state returned from the Authorization Code Grant Flow.")
