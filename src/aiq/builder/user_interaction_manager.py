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
import time
import uuid

from aiq.authentication.exceptions.call_back_exceptions import AuthenticationError
from aiq.authentication.interfaces import OAuthClientManagerBase
from aiq.data_models.authentication import AuthenticationEndpoint
from aiq.data_models.authentication import ConsentPromptMode
from aiq.data_models.interactive import HumanPrompt
from aiq.data_models.interactive import HumanResponse
from aiq.data_models.interactive import InteractionPrompt
from aiq.data_models.interactive import InteractionResponse
from aiq.data_models.interactive import InteractionStatus

logger = logging.getLogger(__name__)


class AIQUserInteractionManager:
    """
    AIQUserInteractionManager is responsible for requesting user input
    at runtime. It delegates the actual prompting to a callback function
    stored in AIQContextState.user_input_callback.

    Type is not imported in __init__ to prevent partial import.
    """

    def __init__(self, context_state: "AIQContextState") -> None:  # noqa: F821
        self._context_state = context_state

    @staticmethod
    async def default_callback_handler(prompt: InteractionPrompt) -> HumanResponse:
        """
        Default callback handler for user input. This is a no-op function
        that simply returns the input text from the Interaction Content
        object.

        Args:
            prompt (InteractionPrompt): The interaction to process.
        """
        raise NotImplementedError("No human prompt callback was registered. Unable to handle requested prompt.")

    @staticmethod
    async def user_auth_callback_default(oauth_client: OAuthClientManagerBase,
                                         consent_prompt_mode: ConsentPromptMode) -> AuthenticationError | None:
        """
        Default callback handler for user authentication strategy. This no-op function raises a NotImplementedError
        error, to indicate that a valid authentication callback was not registered.

        Args:
            oauth_client (OAuthClientBase): The OAuth client to be used for authentication.
            consent_prompt_mode (ConsentPromptMode): The consent prompt mode to be used for browser handling..
        """
        raise NotImplementedError(
            "No authentication callback was registered. Unable to execute authentication strategy.")

    async def prompt_user_input(self, content: HumanPrompt) -> InteractionResponse:
        """
        Ask the user a question and wait for input. This calls out to
        the callback from user_input_callback, which is typically
        set by AIQSessionManager.

        Returns the user's typed-in answer as a string.
        """

        uuid_req = str(uuid.uuid4())
        status = InteractionStatus.IN_PROGRESS
        timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ")
        sys_human_interaction = InteractionPrompt(id=uuid_req, status=status, timestamp=timestamp, content=content)

        resp = await self._context_state.user_input_callback.get()(sys_human_interaction)

        # Rebuild a InteractionResponse object with the response
        timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ")
        status = InteractionStatus.COMPLETED
        sys_human_interaction = InteractionResponse(id=uuid_req, status=status, timestamp=timestamp, content=resp)

        return sys_human_interaction

    async def authenticate_oauth_client(self,
                                        oauth_client: OAuthClientManagerBase,
                                        consent_prompt_handle: ConsentPromptMode) -> AuthenticationError | None:
        # pylint: disable=pointless-statement
        f"""
        Authenticate to an OAuth2.0 API provider using the OAuth2.0 authentication flow with configurable
        consent handling.

        - BROWSER mode: Automatically opens the user's default web browser to the authorization URL. The user
          completes the consent prompt directly in the browser, and upon successful authorization, the browser
          redirects back with an authorization code that is automatically captured and exchanged for access tokens.

        - FRONTEND mode: Logs to the console notifying the user to complete the consent prompt. The user can retreive
        the consent prompt redirect URL by sending a post request to the consent prompt redirect uri:
        {AuthenticationEndpoint.PROMPT_REDIRECT_URI} with the consent prompt key in the request body. The consent
        prompt key is the value of the consent_prompt_key field in the authentication provider configuration.

        Args:
            oauth_client (OAuthClientBase): The OAuth client to be used for authentication.
            consent_prompt_handle (ConsentPromptMode): The consent prompt mode (BROWSER or FRONTEND) that determines
                                                      how the user consent flow is handled.

        Returns:
            AuthenticationError | None: None if authentication succeeds, otherwise returns an AuthenticationError
                                       with details about the failure.
        """
        try:

            await self._context_state.user_auth_callback.get()(oauth_client, consent_prompt_handle)

        except AuthenticationError as e:
            logger.error("Authentication failed while trying to authenticate provider: %s",
                         oauth_client.config_name,
                         exc_info=True)
            return AuthenticationError(error_code="oauth_client_auth_error", message=e)

        return
