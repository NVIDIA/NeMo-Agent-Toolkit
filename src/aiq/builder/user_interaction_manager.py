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

import httpx

from aiq.data_models.api_server import AuthenticatedRequest
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
    async def execute_api_request_default(request: AuthenticatedRequest) -> None:
        """
        Default callback handler for user requests. This no-op function raises a NotImplementedError error, to indicate
        that a valid request callback was not registered.

        Args:
            request (AuthenticatedRequest): The authenticated request to be executed.
        """
        raise NotImplementedError("No request callback was registered. Unable to handle request.")

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

    async def make_api_request(self,
                               url: str,
                               http_method: str,
                               authentication_config_name: str | None = None,
                               headers: dict | None = None,
                               query_params: dict | None = None,
                               body_data: dict | None = None) -> httpx.Response | None:
        """
        Constructs and sends an API request, authenticating the user using OAuth 2.0 or API keys based on the
        credentials specified in the YAML configuration file. If no authentication config is specified, the request
        is sent without authentication. The user is responsible for handling all responses, which are propagated back
        to the user on both successful and unsuccessful requests.

        Args:
            url (str): The base URL to which the request will be sent.
            http_method (str): The HTTP method to use for the request (e.g., "GET", "POST", etc..).
            authentication_config (str | None): The name of the registered authentication config to use for
            making an authenticated request. If None, no authentication will be applied.
            headers (dict | None): Optional dictionary of HTTP headers.
            query_params (dict | None): Optional dictionary of query parameters.
            body_data (dict | None): Optional dictionary representing the request body.

        Returns:
            httpx.Response | None: The successful or unsuccessful response from the API request, or None if an error
            occurs during the authentication process or sending the request.
        """
        from aiq.authentication.credentials_manager import _CredentialsManager

        try:
            authenticated_request: AuthenticatedRequest = AuthenticatedRequest(
                url_path=url,
                method=http_method,
                authentication_config_name=authentication_config_name,
                authentication_config=_CredentialsManager().get_authentication_config(authentication_config_name),
                headers=headers,
                query_params=query_params,
                body_data=body_data)

            response: httpx.Response | None = await self._context_state.user_request_callback.get()(
                authenticated_request)

        except Exception as e:
            logger.error("Error while making API request: %s", str(e), exc_info=True)
            return None

        return response
