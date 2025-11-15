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
Simple example demonstrating /evaluate/item endpoint WITHOUT trajectory processing.

This is the simpler, faster version suitable for most evaluators that only need:
- Input question
- Expected output
- Actual output

Suitable for: accuracy, groundedness, relevance evaluators
NOT suitable for: trajectory_accuracy (use evaluate_single_item.py instead)

SETUP (REQUIRED):
-----------------
1. Set your API key:
   export NVIDIA_API_KEY=<YOUR_API_KEY>

2. Start the server in one terminal:
   nat serve --config_file examples/evaluation_and_profiling/simple_web_query_eval/configs/eval_config.yml

3. In another terminal, run this script:
   python examples/evaluation_and_profiling/simple_web_query_eval/scripts/evaluate_single_item_simple.py

WHAT IT DOES:
-------------
- Sends a question to /generate/full endpoint (without intermediate steps)
- Captures the agent's response
- Evaluates using /evaluate/item endpoint
- Displays the evaluation score and reasoning

CUSTOMIZE:
----------
Edit the CONFIGURATION section below to:
- Change the server URL (if not running on localhost:8000)
- Use a different evaluator (accuracy, groundedness, relevance)
- Test with different questions
"""

import asyncio
import json
import logging
import sys

import aiohttp

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURATION - Customize these for your setup
# ============================================================================

BASE_URL = "http://localhost:8000"
"""Base URL of the NAT server. Change if running on different host/port."""

EVALUATOR_NAME = "accuracy"
"""
Evaluator to use. Must match an evaluator name in eval_config.yml.
Good options for simple evaluation: accuracy, groundedness, relevance
(Don't use trajectory_accuracy here - it needs the full script with trajectory)
"""

INPUT_MESSAGE = "What is LangSmith?"
"""The question to ask the agent."""

EXPECTED_OUTPUT = "LangSmith is a platform for building production-grade LLM applications."
"""The expected/reference answer for evaluation."""


async def run_and_evaluate_simple(base_url: str, input_message: str, expected_output: str, evaluator_name: str):
    """
    Simple workflow evaluation without trajectory processing.

    Args:
        base_url: Base URL of the NAT server
        input_message: Question to ask
        expected_output: Expected answer
        evaluator_name: Name of evaluator to use
    """

    async with aiohttp.ClientSession() as session:
        # ========================================================================
        # STEP 1: Run workflow (without intermediate steps)
        # ========================================================================
        logger.info("=" * 70)
        logger.info("STEP 1: Running workflow (simple mode - no trajectory)")
        logger.info("=" * 70)
        logger.info("Question: %s", input_message)

        # Use filter_steps=none to suppress intermediate steps for speed
        endpoint = f"{base_url}/generate/full?filter_steps=none"
        payload = {"input_message": input_message}

        final_response = None

        try:
            async with session.post(endpoint, json=payload) as response:
                response.raise_for_status()

                # Process streaming response - only looking for final output
                async for line in response.content:
                    line = line.decode('utf-8').strip()
                    if not line:
                        continue

                    # Only parse data lines (no intermediate_data in this mode)
                    if line.startswith("data: "):
                        try:
                            chunk_data = json.loads(line[6:])  # Skip "data: " prefix
                            if chunk_data.get("value"):
                                final_response = chunk_data.get("value")
                        except json.JSONDecodeError as e:
                            logger.exception("Failed to parse response: %s", e)
                            continue

        except aiohttp.ClientError as e:
            logger.exception("Request failed: %s", e)
            logger.error("\n❌ ERROR: Could not connect to server at %s", base_url)
            logger.error("Make sure the server is running with:")
            logger.error("  nat serve \
                    --config_file \
                        examples/evaluation_and_profiling/simple_web_query_eval/configs/eval_config.yml")
            return None

        logger.info("")
        logger.info("✓ Workflow completed")
        logger.info("  Output: %s", final_response)

        # ========================================================================
        # STEP 2: Evaluate the result
        # ========================================================================
        logger.info("")
        logger.info("=" * 70)
        logger.info("STEP 2: Evaluating result")
        logger.info("=" * 70)
        logger.info("Evaluator: %s", evaluator_name)

        eval_payload = {
            "evaluator_name": evaluator_name,
            "item": {
                "id": "test_item_1",
                "input_obj": input_message,
                "expected_output_obj": expected_output,
                "output_obj": final_response,
                "trajectory": [],  # Empty - not needed for most evaluators
                "expected_trajectory": [],
                "full_dataset_entry": {}
            }
        }

        try:
            eval_endpoint = f"{base_url}/evaluate/item"
            async with session.post(eval_endpoint, json=eval_payload) as response:
                if response.status == 404:
                    error_detail = await response.json()
                    logger.error("\n❌ ERROR: Evaluator not found")
                    logger.error("  %s", error_detail.get('detail', 'Unknown error'))
                    logger.error("\nMake sure '%s' is configured in eval_config.yml", evaluator_name)
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
                logger.info("Success: %s", result['success'])

                if result['success']:
                    eval_result = result['result']
                    logger.info("Score: %s", eval_result['score'])
                    logger.info("\nReasoning:")
                    logger.info(json.dumps(eval_result['reasoning'], indent=2))
                    logger.info("=" * 70)
                    logger.info("\n✓ Evaluation completed successfully!")
                else:
                    logger.error("\n❌ Evaluation failed: %s", result['error'])
                    logger.info("=" * 70)

                return result

        except aiohttp.ClientError as e:
            logger.exception("Evaluation request failed: %s", e)
            logger.error("\n❌ ERROR: Failed to evaluate item")
            return None


async def main():
    """Main entry point."""
    print("\n" + "=" * 70)
    print("EVALUATE SINGLE ITEM - Simple Mode (No Trajectory)")
    print("=" * 70)
    print(f"Server:    {BASE_URL}")
    print(f"Evaluator: {EVALUATOR_NAME}")
    print(f"Question:  {INPUT_MESSAGE}")
    print("=" * 70)
    print()

    result = await run_and_evaluate_simple(base_url=BASE_URL,
                                           input_message=INPUT_MESSAGE,
                                           expected_output=EXPECTED_OUTPUT,
                                           evaluator_name=EVALUATOR_NAME)

    if result:
        return 0
    else:
        print("\n❌ Failed to complete evaluation")
        print("\nTroubleshooting:")
        print("1. Ensure the server is running")
        print("2. Check that your NVIDIA_API_KEY is set")
        print("3. Verify the evaluator name matches one in eval_config.yml")
        print("4. Don't use trajectory_accuracy with this script (use evaluate_single_item.py)")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
