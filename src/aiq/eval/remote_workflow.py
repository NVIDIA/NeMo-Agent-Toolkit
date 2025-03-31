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

import asyncio
import logging

import aiohttp
from pydantic import ValidationError
from tqdm import tqdm

from aiq.data_models.api_server import AIQGenerateResponse
from aiq.data_models.evaluate import EvalConfig
from aiq.eval.config import EvaluationRunConfig
from aiq.eval.evaluator.evaluator_model import EvalInput
from aiq.eval.evaluator.evaluator_model import EvalInputItem

logger = logging.getLogger(__name__)


class EvaluationRemoteWorkflowHandler:

    def __init__(self, config: EvaluationRunConfig, eval_config: EvalConfig):
        self.config = config
        self.eval_config = eval_config

        # Run metadata
        self.semaphore = asyncio.Semaphore(self.eval_config.general.max_concurrency)

    async def run_workflow_remote_single(self, session: aiohttp.ClientSession, item: EvalInputItem):
        """
        Sends a single input to the endpoint hosting the workflow and retrieves the response.
        """
        question = item.input_obj
        # generate request is a dict with a single key "input_message"
        payload = {"input_message": question}
        try:
            async with session.post(self.config.endpoint, json=payload) as response:
                response.raise_for_status()  # Raise an exception for HTTP errors
                json_response = await response.json()
        except aiohttp.ClientError as e:
            # Handle connection or HTTP-related errors
            logger.error("Request failed for question %s: %s", question, e)
            item.output_obj = None
            item.trajectory = []
            return

        try:
            generate_response = AIQGenerateResponse.model_validate(json_response)
        except ValidationError as e:
            logger.error("Validation failed for question: %s\nResponse: %s\nError: %s", question, json_response, e)
            item.output_obj = None
            item.trajectory = []
            return

        # Extract and fill the item with the response and intermediate steps
        item.output_obj = generate_response.output
        item.trajectory = generate_response.intermediate_steps
        return

    async def run_workflow_remote_with_limits(self, session: aiohttp.ClientSession, item: EvalInputItem, pbar: tqdm):
        """
        Sends limited number of concurrent requests to a remote workflow and retrieves responses.
        """
        async with self.semaphore:
            await self.run_workflow_remote_single(session=session, item=item)
            pbar.update(1)

    async def run_workflow_remote(self, eval_input: EvalInput) -> EvalInput:
        """
        Sends inputs to a workflow hosted on a remote endpoint.
        """
        timeout = aiohttp.ClientTimeout(total=self.config.endpoint_timeout)
        try:
            pbar = tqdm(total=len(eval_input.eval_input_items), desc="Running workflow", unit="item")
            async with aiohttp.ClientSession(timeout=timeout) as session:
                # get the questions from the eval_input
                tasks = [
                    self.run_workflow_remote_with_limits(session, item, pbar) for item in eval_input.eval_input_items
                ]
                await asyncio.gather(*tasks)

        finally:
            pbar.close()

        return eval_input
