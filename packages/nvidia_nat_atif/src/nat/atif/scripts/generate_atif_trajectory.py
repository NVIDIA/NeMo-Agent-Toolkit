# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Run a NeMo Agent Toolkit workflow and export the trajectory as ATIF JSON.

This script loads any NAT workflow from a YAML config, executes it with a given
input, captures all IntermediateStep events, converts them to an ATIF v1.6
trajectory, and writes the result to a JSON file.

Prerequisites:
    - The workflow's package must be installed (for example,
      ``pip install -e examples/getting_started/simple_calculator``).
    - An appropriate API key must be set (for example, ``NVIDIA_API_KEY``).

Usage (from repo root):
    python -m nat.atif.scripts.generate_atif_trajectory \\
        --config examples/getting_started/simple_calculator/src/nat_simple_calculator/configs/config.yml \\
        --input "What is 7 * 8?" \\
        -o atif_output.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


async def run_and_capture_atif(
    config_file: Path,
    question: str,
    session_id: str | None = None,
) -> dict:
    """Load a workflow, run it, and return the ATIF trajectory as a dict."""
    from nat.builder.context import Context
    from nat.data_models.intermediate_step import IntermediateStep
    from nat.runtime.loader import load_workflow
    from nat.utils.atif_converter import IntermediateStepToATIFConverter

    collected_steps: list[IntermediateStep] = []
    done_event = asyncio.Event()

    async with load_workflow(config_file) as workflow:
        async with workflow.run(question) as runner:
            context = Context.get()

            def on_next(step: IntermediateStep) -> None:
                collected_steps.append(step)

            def on_error(exc: Exception) -> None:
                logger.error("IntermediateStep stream error: %s", exc)
                done_event.set()

            def on_complete() -> None:
                done_event.set()

            context.intermediate_step_manager.subscribe(
                on_next=on_next,
                on_error=on_error,
                on_complete=on_complete,
            )

            result = await runner.result(to_type=str)
            await done_event.wait()

    logger.info("Collected %d intermediate steps", len(collected_steps))
    logger.info("Workflow result: %s", result)

    converter = IntermediateStepToATIFConverter()
    trajectory = converter.convert(
        collected_steps,
        session_id=session_id,
    )

    return trajectory.to_json_dict()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a NAT workflow and export the ATIF trajectory as JSON.", )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the workflow YAML config file.",
    )
    parser.add_argument(
        "--input",
        type=str,
        default="What is 12 * 15 + 8?",
        help="The question to send to the workflow.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="atif_output.json",
        help="Output JSON file path (default: atif_output.json).",
    )
    parser.add_argument(
        "--session-id",
        type=str,
        default=None,
        help="Explicit session ID for the trajectory. Auto-generated if omitted.",
    )
    args = parser.parse_args()

    config_file = Path(args.config)
    if not config_file.exists():
        print(f"Config not found: {config_file}", file=sys.stderr)
        sys.exit(1)

    output_path = Path(args.output)

    print(f"Config:   {config_file}")
    print(f"Question: {args.input}")
    print(f"Output:   {output_path}")
    print()

    traj_dict = asyncio.run(
        run_and_capture_atif(
            config_file=config_file,
            question=args.input,
            session_id=args.session_id,
        ))

    output_path.write_text(json.dumps(traj_dict, indent=2) + "\n")
    print(f"\nATIF trajectory written to: {output_path}")
    print(f"Steps: {len(traj_dict.get('steps', []))}")
    if traj_dict.get("final_metrics"):
        fm = traj_dict["final_metrics"]
        print(f"Total prompt tokens:     {fm.get('total_prompt_tokens', 'N/A')}")
        print(f"Total completion tokens:  {fm.get('total_completion_tokens', 'N/A')}")
        print(f"Total agent steps:        {fm.get('total_steps', 'N/A')}")


if __name__ == "__main__":
    main()
