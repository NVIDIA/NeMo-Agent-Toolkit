#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
Create a filtered test subset from the full Agent Leaderboard v2 dataset.

This script selects a few scenarios for quick validation testing.
"""

import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_test_subset(input_file: Path, output_file: Path, num_scenarios: int = 3) -> None:
    """
    Create a test subset with a limited number of scenarios.

    Args:
        input_file: Full dataset file
        output_file: Output file for test subset
        num_scenarios: Number of scenarios to include
    """
    logger.info("Loading full dataset from %s", input_file)

    with open(input_file) as f:
        full_dataset = json.load(f)

    logger.info("Loaded %d scenarios from full dataset", len(full_dataset))

    # Select first N scenarios
    test_subset = full_dataset[:num_scenarios]

    logger.info("Created test subset with %d scenarios", len(test_subset))

    # Save test subset
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(test_subset, f, indent=2)

    logger.info("Saved test subset to %s", output_file)

    # Print summary
    for i, scenario in enumerate(test_subset):
        logger.info(
            "Scenario %d: id=%s, goals=%d, tools=%d, expected_calls=%d",
            i + 1,
            scenario.get("id"),
            len(scenario.get("user_goals", [])),
            len(scenario.get("available_tools", [])),
            len(scenario.get("expected_tool_calls", [])),
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Create test subset from Agent Leaderboard v2 dataset")
    parser.add_argument(
        "--input-file",
        type=Path,
        default=Path(__file__).parent.parent / "data" / "agent_leaderboard_v2_banking.json",
        help="Input dataset file",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=Path(__file__).parent.parent / "data" / "agent_leaderboard_v2_test_subset.json",
        help="Output test subset file",
    )
    parser.add_argument(
        "--num-scenarios",
        type=int,
        default=3,
        help="Number of scenarios to include",
    )

    args = parser.parse_args()

    create_test_subset(args.input_file, args.output_file, args.num_scenarios)
