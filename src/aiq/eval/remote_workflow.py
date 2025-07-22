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
import json
import logging

import aiohttp
from pydantic import ValidationError
from tqdm import tqdm

from aiq.data_models.api_server import AIQResponseIntermediateStep
from aiq.data_models.intermediate_step import IntermediateStep
from aiq.data_models.intermediate_step import IntermediateStepPayload
from aiq.data_models.invocation_node import InvocationNode
from aiq.eval.config import EvaluationRunConfig
from aiq.eval.evaluator.evaluator_model import EvalInput
from aiq.eval.evaluator.evaluator_model import EvalInputItem

logger = logging.getLogger(__name__)

# Constants for streaming response prefixes
DATA_PREFIX = "data: "
INTERMEDIATE_DATA_PREFIX = "intermediate_data: "


class EvaluationRemoteWorkflowHandler:

    def __init__(self, config: EvaluationRunConfig, max_concurrency: int):
        self.config = config
        # Run metadata
        self.semaphore = asyncio.Semaphore(max_concurrency)

    async def run_workflow_remote_single(self, session: aiohttp.ClientSession, item: EvalInputItem):
        """
        Sends a single input to the endpoint hosting the workflow and retrieves the response.
        """
        question = item.input_obj
        # generate request format
        payload = {"input_message": question}

        try:
            # Use the streaming endpoint
            endpoint = f"{self.config.endpoint}/generate/full"
            async with session.post(endpoint, json=payload) as response:
                response.raise_for_status()  # Raise an exception for HTTP errors

                # Initialize variables to store the response
                final_response = None
                intermediate_steps = []

                # Process the streaming response
                async for line in response.content:
                    line = line.decode('utf-8').strip()
                    if not line:
                        continue

                    if line.startswith(DATA_PREFIX):
                        # This is a generate response chunk
                        try:
                            chunk_data = json.loads(line[len(DATA_PREFIX):])
                            if chunk_data.get("value"):
                                final_response = chunk_data.get("value")
                        except json.JSONDecodeError as e:
                            logger.error("Failed to parse generate response chunk: %s", e)
                            continue
                    elif line.startswith(INTERMEDIATE_DATA_PREFIX):
                        # This is an intermediate step
                        try:
                            step_data = json.loads(line[len(INTERMEDIATE_DATA_PREFIX):])
                            response_intermediate = AIQResponseIntermediateStep.model_validate(step_data)
                            # The payload is expected to be IntermediateStepPayload
                            payload = IntermediateStepPayload.model_validate_json(response_intermediate.payload)
                            intermediate_step = IntermediateStep(parent_id="remote",
                                                                 function_ancestry=InvocationNode(
                                                                     function_name=payload.name or "remote_function",
                                                                     function_id=payload.UUID or "remote_function_id"),
                                                                 payload=payload)
                            intermediate_steps.append(intermediate_step)
                        except (json.JSONDecodeError, ValidationError) as e:
                            logger.error("Failed to parse intermediate step: %s", e)
                            continue

        except aiohttp.ClientError as e:
            # Handle connection or HTTP-related errors
            logger.error("Request failed for question %s: %s", question, e)
            item.output_obj = None
            item.trajectory = []
            return

        # Extract and fill the item with the response and intermediate steps
        item.output_obj = final_response
        item.trajectory = intermediate_steps
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
