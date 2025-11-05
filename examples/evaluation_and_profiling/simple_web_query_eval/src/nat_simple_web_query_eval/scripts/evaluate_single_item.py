#!/usr/bin/env python3
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
"""
Demonstrate /evaluate_item endpoint WITH trajectory processing (full version).

This is the complete version that captures and processes intermediate steps (trajectory).
Use this when you need trajectory information for your evaluator.

Suitable for: trajectory_accuracy evaluator
For simpler evaluations: Use evaluate_single_item_simple.py instead

This script shows how to evaluate a single workflow execution by:
1. Running a query via /generate/full endpoint
2. Parsing the streaming response (output + intermediate steps/trajectory)
3. Evaluating the result via /evaluate_item endpoint

SETUP (REQUIRED):
-----------------
Before running this script, you must start the workflow server with evaluators configured.

1. Set your API key:
   export NVIDIA_API_KEY=<YOUR_API_KEY>

2. Start the server in one terminal:
   nat serve --config_file examples/evaluation_and_profiling/simple_web_query_eval/configs/eval_config.yml

3. In another terminal, run this script:
   python examples/evaluation_and_profiling/simple_web_query_eval/scripts/evaluate_single_item.py

WHAT IT DOES:
-------------
- Sends a question about LangSmith to the /generate/full endpoint
- Captures the agent's response and intermediate steps (trajectory)
- Evaluates the response using the "accuracy" evaluator
- Displays the evaluation score and reasoning

CUSTOMIZE:
----------
Edit the CONFIGURATION section below to:
- Change the server URL (if not running on localhost:8000)
- Use a different evaluator (accuracy, groundedness, relevance, trajectory_accuracy)
- Test with different questions
- Modify expected answers for testing
"""

import asyncio
import json
import logging
import sys

import aiohttp
from pydantic import ValidationError

# Import NAT data models (same pattern as remote_workflow.py)
try:
    from nat.data_models.api_server import ResponseIntermediateStep
    from nat.data_models.intermediate_step import IntermediateStep
    from nat.data_models.intermediate_step import IntermediateStepPayload
    from nat.data_models.invocation_node import InvocationNode
except ImportError as e:
    print("Error: NAT modules not found. Make sure you're running from NAT environment.")
    print(f"Import error: {e}")
    print("\nTo install: pip install nvidia-nat")
    sys.exit(1)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURATION - Customize these for your setup
# ============================================================================

BASE_URL = "http://localhost:8000"
"""Base URL of the NAT server. Change if running on different host/port."""

EVALUATOR_NAME = "trajectory_accuracy"
"""
Evaluator to use. Must match an evaluator name in eval_config.yml.
This script is designed for trajectory-based evaluators.
For simpler evaluators (accuracy, groundedness, relevance), use evaluate_single_item_simple.py
"""

INPUT_MESSAGE = "What is LangSmith?"
"""The question to ask the agent."""

EXPECTED_OUTPUT = "LangSmith is a platform for building production-grade LLM applications."
"""
The expected/reference answer for evaluation.
The evaluator will compare the agent's actual response against this.
"""

# ============================================================================
# Constants (from nat/eval/remote_workflow.py)
# ============================================================================

DATA_PREFIX = "data: "
INTERMEDIATE_DATA_PREFIX = "intermediate_data: "


# ============================================================================
# Main Implementation
# ============================================================================
async def run_workflow_and_evaluate(base_url: str, input_message: str, expected_output: str, evaluator_name: str):
    """
    Run a workflow query and evaluate the result.

    This follows the same pattern as EvaluationRemoteWorkflowHandler.run_workflow_remote_single
    from nat/eval/remote_workflow.py.

    Args:
        base_url: Base URL of the NAT server
        input_message: Question to ask the workflow
        expected_output: Expected answer for evaluation
        evaluator_name: Name of evaluator to use

    Returns:
        dict: Evaluation result containing success status, score, and reasoning
    """

    async with aiohttp.ClientSession() as session:
        # ========================================================================
        # STEP 1: Run the workflow via /generate/full
        # ========================================================================
        logger.info("=" * 70)
        logger.info("STEP 1: Running workflow")
        logger.info("=" * 70)
        logger.info(f"Question: {input_message}")

        payload = {"input_message": input_message}
        endpoint = f"{base_url}/generate/full"

        final_response = None
        intermediate_steps = []

        try:
            async with session.post(endpoint, json=payload) as response:
                response.raise_for_status()

                # Process streaming response (following remote_workflow.py pattern)
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
                            logger.exception("Failed to parse generate response chunk: %s", e)
                            continue

                    elif line.startswith(INTERMEDIATE_DATA_PREFIX):
                        # This is an intermediate step (trajectory)
                        # Parse exactly as done in remote_workflow.py lines 79-91
                        try:
                            step_data = json.loads(line[len(INTERMEDIATE_DATA_PREFIX):])
                            response_intermediate = ResponseIntermediateStep.model_validate(step_data)
                            # The payload is expected to be IntermediateStepPayload
                            payload_obj = IntermediateStepPayload.model_validate_json(response_intermediate.payload)
                            intermediate_step = IntermediateStep(
                                parent_id="remote",
                                function_ancestry=InvocationNode(function_name=payload_obj.name or "remote_function",
                                                                 function_id=payload_obj.UUID or "remote_function_id"),
                                payload=payload_obj)
                            intermediate_steps.append(intermediate_step)
                        except (json.JSONDecodeError, ValidationError) as e:
                            logger.exception("Failed to parse intermediate step: %s", e)
                            continue

        except aiohttp.ClientError as e:
            logger.exception("Request failed: %s", e)
            logger.error("\n❌ ERROR: Could not connect to server at %s", base_url)
            logger.error("Make sure the server is running with:")
            logger.error(
                "  nat serve --config_file examples/evaluation_and_profiling/simple_web_query_eval/configs/eval_config.yml"
            )
            return None

        logger.info("")
        logger.info("✓ Workflow completed")
        logger.info(f"  Output: {final_response}")
        logger.info(f"  Captured {len(intermediate_steps)} intermediate steps")

        # ========================================================================
        # STEP 2: Evaluate the result via /evaluate_item
        # ========================================================================
        logger.info("")
        logger.info("=" * 70)
        logger.info("STEP 2: Evaluating result")
        logger.info("=" * 70)
        logger.info(f"Evaluator: {evaluator_name}")

        # Convert IntermediateStep objects to dicts for JSON serialization
        trajectory_dicts = [step.model_dump() for step in intermediate_steps]

        eval_payload = {
            "evaluator_name": evaluator_name,
            "item": {
                "id": "test_item_1",
                "input_obj": input_message,
                "expected_output_obj": expected_output,
                "output_obj": final_response,
                "trajectory": trajectory_dicts,
                "expected_trajectory": [],
                "full_dataset_entry": {}
            }
        }

        try:
            eval_endpoint = f"{base_url}/evaluate_item"
            async with session.post(eval_endpoint, json=eval_payload) as response:
                if response.status == 404:
                    error_detail = await response.json()
                    logger.error("\n❌ ERROR: Evaluator not found")
                    logger.error(f"  {error_detail.get('detail', 'Unknown error')}")
                    logger.error(f"\nMake sure '{evaluator_name}' is configured in eval_config.yml")
                    logger.error("Available evaluators: accuracy, groundedness, relevance, trajectory_accuracy")
                    return None

                response.raise_for_status()
                result = await response.json()

                # ================================================================
                # STEP 3: Display results
                # ================================================================
                logger.info("")
                logger.info("=" * 70)
                logger.info("EVALUATION RESULTS")
                logger.info("=" * 70)
                logger.info(f"Success: {result['success']}")

                if result['success']:
                    eval_result = result['result']
                    logger.info(f"Score: {eval_result['score']}")
                    logger.info(f"\nReasoning:")
                    logger.info(json.dumps(eval_result['reasoning'], indent=2))
                    logger.info("=" * 70)
                    logger.info("\n✓ Evaluation completed successfully!")
                else:
                    logger.error(f"\n❌ Evaluation failed: {result['error']}")
                    logger.info("=" * 70)

                return result

        except aiohttp.ClientError as e:
            logger.exception("Evaluation request failed: %s", e)
            logger.error("\n❌ ERROR: Failed to evaluate item")
            logger.error("Make sure the /evaluate_item endpoint is available in your server")
            return None


async def main():
    """Main entry point."""
    print("\n" + "=" * 70)
    print("EVALUATE SINGLE ITEM - Demonstration Script")
    print("=" * 70)
    print(f"Server:    {BASE_URL}")
    print(f"Evaluator: {EVALUATOR_NAME}")
    print(f"Question:  {INPUT_MESSAGE}")
    print("=" * 70)
    print()

    result = await run_workflow_and_evaluate(base_url=BASE_URL,
                                             input_message=INPUT_MESSAGE,
                                             expected_output=EXPECTED_OUTPUT,
                                             evaluator_name=EVALUATOR_NAME)

    if result:
        return 0
    else:
        print("\n❌ Failed to complete evaluation")
        print("\nTroubleshooting:")
        print("1. Ensure the server is running:")
        print(
            "   nat serve --config_file examples/evaluation_and_profiling/simple_web_query_eval/configs/eval_config.yml"
        )
        print("2. Check that your NVIDIA_API_KEY is set")
        print("3. Verify the evaluator name matches one in eval_config.yml")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
