# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from starlette.requests import Request

from a2a.auth.user import User
from a2a.server.apps.jsonrpc.jsonrpc_app import DefaultCallContextBuilder
from a2a.server.context import ServerCallContext
from nat.runtime.user_manager import UserManager

logger = logging.getLogger(__name__)


class AuthenticatedUser(User):
    """An A2A user carrying NAT's user ID for an authenticated request."""

    def __init__(self, user_id: str):
        self._user_id = user_id

    @property
    def is_authenticated(self) -> bool:
        return True

    @property
    def user_name(self) -> str:
        return self._user_id


class NATCallContextBuilder(DefaultCallContextBuilder):
    """Builds the A2A call context from a request NAT has already authenticated.

    The default builder reads `request.user`, which only exists when Starlette's
    `AuthenticationMiddleware` is installed. NAT authenticates with its own
    `OAuth2ValidationMiddleware` instead, so without this the call context is
    always unauthenticated and a per-user workflow has no user to build for.
    """

    def build(self, request: Request) -> ServerCallContext:
        context = super().build(request)

        # Only trust a request OAuth2ValidationMiddleware has already verified.
        if getattr(request.state, "oauth_user", None) is None:
            return context

        try:
            user_info = UserManager.extract_user_from_connection(request)
        except ValueError:
            # A credential that cannot be resolved leaves the context unauthenticated
            # rather than failing the request here.
            logger.warning("Could not resolve a user identity from an authenticated request", exc_info=True)
            return context

        if user_info is None:
            return context

        context.user = AuthenticatedUser(user_info.get_user_id())
        return context
