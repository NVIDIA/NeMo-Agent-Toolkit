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

import aiohttp
from tqdm.asyncio import tqdm

from aiq.data_models.evaluate import EvalConfig
from aiq.eval.config import EvaluationRunConfig

logger = logging.getLogger(__name__)


class EvaluationRemoteWorkflowHandler:

    def __init__(self, config: EvaluationRunConfig, eval_config: EvalConfig):
        import asyncio
        self.config = config
        self.eval_config = eval_config

        # Run metadata
        self.semaphore = asyncio.Semaphore(self.eval_config.max_concurrency)

    async def run_workflow_remote_single(self, session: aiohttp.ClientSession, question: str) -> dict:
        """
        Sends a single question to the endpoint hosting the worflow and retrieves the response.
        """
        from pydantic import ValidationError

        from aiq.data_models.api_server import AIQChatRequest
        from aiq.data_models.api_server import AIQChatResponse
        from aiq.data_models.api_server import Message

        workflow_llm = self.eval_config.workflow_llm
        chat_request = AIQChatRequest(model=workflow_llm.model,
                                      temperature=workflow_llm.temperature,
                                      top_p=workflow_llm.top_p,
                                      max_tokens=workflow_llm.tokens,
                                      messages=[Message(role="user", content=question)])
        payload = chat_request.model_dump()
        try:
            async with session.post(self.config.endpoint, json=payload) as response:
                response.raise_for_status()  # Raise an exception for HTTP errors
                json_response = await response.json()
        except aiohttp.ClientError as e:
            # Handle connection or HTTP-related errors
            logger.error("Request failed for question %s: %s", question, e)
            return {"error": str(e), "question": question}

        try:
            chat_response = AIQChatResponse.model_validate(json_response)
        except ValidationError as e:
            logger.error("Response validation failed: %s", e)
            return {"error": "Response validation failed", "details": str(e)}

        # Extract and return the content
        if not chat_response.choices:
            logger.error("Received empty choices from the endpoint for question '%s': %s", question, json_response)
            return {"error": "Empty response choices", "question": question}

        return {"response": chat_response.choices[-1].message.content}

    async def run_workflow_remote_with_limits(self, session: aiohttp.ClientSession, question: str) -> dict:
        """
        Sends limited number of concurrent requests to a remote workflow and retrieves responses.
        """
        async with self.semaphore:
            return await self.run_workflow_remote_single(session=session, question=question)

    async def run_workflow_remote(self, questions: list[str]) -> list:
        """
        Sends question to a workflow hosted on a remote endpoint.
        """
        timeout = aiohttp.ClientTimeout(total=self.config.endpoint_timeout)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            tasks = [self.run_workflow_remote_with_limits(session=session, question=question) for question in questions]
            return await tqdm.gather(*tasks)
