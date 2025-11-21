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
Red Teaming Evaluation Example for Simple Calculator

This script demonstrates how to use the RedTeamingEvaluationRunner to test
a workflow with different function intercept configurations.

The intercept scenarios are defined in a JSON file where each entry specifies:
- Which intercept to apply (or null for baseline)
- Which function or function_group to apply it to
- What payload value to use

Usage:
    python run_redteam_eval.py
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add current directory to path to ensure local imports work
current_dir = Path(__file__).parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

# Import middleware module to ensure registration decorator executes
from calculator_middleware import register  # noqa: E402, F401

# Import NAT components after path setup
from nat.eval.config import EvaluationRunConfig  # noqa: E402
from nat.eval.runners.red_team_eval_runner.red_team_eval_config import RedTeamingEvaluationConfig  # noqa: E402
from nat.eval.runners.red_team_eval_runner.red_team_eval_runner import RedTeamingEvaluationRunner  # noqa: E402

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def main():
    """Run red teaming evaluation across multiple intercept scenarios."""

    # Define paths
    current_dir = Path(__file__).parent
    base_config_path = current_dir / "configs" / "base_workflow.yml"
    red_team_scenarios_file = current_dir / "data" / "red_team_scenarios.json"
    dataset_path = current_dir / "data" / "calculator_test_dataset.json"

    # Create base evaluation config
    base_eval_config = EvaluationRunConfig(
        config_file=base_config_path,
        dataset=str(dataset_path),
        skip_completed_entries=False,
        write_output=False,
    )

    # Create red teaming evaluation config
    redteam_config = RedTeamingEvaluationConfig(
        base_evaluation_config=base_eval_config,
        red_team_scenarios_file=red_team_scenarios_file
    )

    # Create runner and execute all scenarios
    logger.info("Starting red teaming evaluation...")
    runner = RedTeamingEvaluationRunner(redteam_config)
    results = await runner.run_all()

    # Print summary of results
    logger.info("\n" + "=" * 80)
    logger.info("RED TEAMING EVALUATION SUMMARY")
    logger.info("=" * 80)

    for scenario_id, output in results.items():
        logger.info(f"\nScenario: {scenario_id}")
        logger.info(f"  Workflow Output: {output.workflow_output_file}")
        logger.info(f"  Evaluator Outputs: {len(output.evaluator_output_files)} files")
        logger.info(f"  Workflow Interrupted: {output.workflow_interrupted}")

        # Print evaluation results
        for evaluator_name, eval_output in output.evaluation_results:
            logger.info(f"  Evaluator '{evaluator_name}':")
            logger.info(f"    Score: {eval_output.average_score}")

    logger.info("\n" + "=" * 80)
    logger.info("Red teaming evaluation completed!")


if __name__ == "__main__":
    asyncio.run(main())
