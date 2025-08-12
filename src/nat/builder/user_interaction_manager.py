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

from nat.data_models.interactive import HumanPrompt
from nat.data_models.interactive import HumanResponse
from nat.data_models.interactive import InteractionPrompt
from nat.data_models.interactive import InteractionResponse
from nat.data_models.interactive import InteractionStatus

logger = logging.getLogger(__name__)


class UserInteractionManager:
    """
    UserInteractionManager is responsible for requesting user input
    at runtime. It delegates the actual prompting to a callback function
    stored in ContextState.user_input_callback.

    Type is not imported in __init__ to prevent partial import.
    """

    def __init__(self, context_state: "ContextState") -> None:  # noqa: F821
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
        set by SessionManager.

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


# Compatibility aliases with previous releases
AIQUserInteractionManager = UserInteractionManager
