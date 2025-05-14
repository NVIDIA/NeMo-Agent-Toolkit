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

from aiq.authentication.request_manager import RequestManager
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
                               authentication_provider: str | None = None,
                               headers: dict | None = None,
                               params: dict | None = None,
                               data: dict | None = None) -> httpx.Response | None:
        """
        Args: # TODO EE: Update doc strings and error handling.
            url (str | httpx.URL): The base URL to which the request will be sent.
            http_method (str | HTTPMethod): The HTTP method to use for the request (e.g., "GET", "POST").
            authentication_provider( str | None): The name of the registered authentication provider to make an
            authenticated request.
            headers (dict | None): Optional dictionary of HTTP headers.
            query_params (dict | None): Optional dictionary of query parameters.
            data (dict | None): Optional dictionary representing the request body.
        Returns:
            httpx.Response | None: _description_
        """

        request = RequestManager()
        response: httpx.Response | None = None

        response = await request._send_request(
            url=url,  # TODO EE: Need to use callback and remove the command to determine of its console / server etc...
            http_method=http_method,
            authentication_provider=authentication_provider,
            headers=headers,
            query_params=params,
            data=data)

        return response
