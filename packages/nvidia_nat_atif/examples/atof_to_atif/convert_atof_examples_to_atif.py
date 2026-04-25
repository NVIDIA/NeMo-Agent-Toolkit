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

"""Convert ATOF v0.1 JSONL examples to ATIF trajectories.

Reads each example JSONL file produced by ``generate_atof_examples.py``
(EXMP-01 tier-1, EXMP-02 tier-2) and writes the resulting ATIF trajectory
as formatted JSON.

Uses ``nat.atof.scripts.atof_to_atif_converter.convert_file`` — the v0.1
converter dispatches on ``(kind, scope_category, category)`` and reads
category-specific fields from the ``category_profile`` sub-object
(``category_profile.model_name``, ``category_profile.tool_call_id``).

Usage:
    python convert_atof_examples_to_atif.py [--input-dir DIR] [--output-dir DIR]
"""

from __future__ import annotations

import argparse
from pathlib import Path

from nat.atof.scripts.atof_to_atif_converter import convert_file

EXAMPLES_DIR = Path(__file__).parent
OUTPUT_DIR = EXAMPLES_DIR / "output"

EXAMPLES = [
    "exmp01_atof.jsonl",
    "exmp02_atof.jsonl",
    "exmp03_atof.jsonl",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert ATOF v0.1 JSONL to ATIF JSON")
    parser.add_argument("--input-dir", type=Path, default=OUTPUT_DIR, help="Directory with JSONL files")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR, help="Output directory for ATIF JSON")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    for filename in EXAMPLES:
        input_path = args.input_dir / filename
        if not input_path.exists():
            print(f"Skipping {filename} (not found)")
            continue

        # Symmetric naming: exmpNN_atof.jsonl -> exmpNN_atif.json
        output_name = filename.replace("_atof.jsonl", "_atif.json")
        output_path = args.output_dir / output_name

        trajectory = convert_file(input_path, output_path)

        print(f"{filename} -> {output_name}")
        print(f"  Steps: {len(trajectory.steps)}")
        print(f"  Agent: {trajectory.agent.name}")
        for step in trajectory.steps:
            tc = len(step.tool_calls) if step.tool_calls else 0
            obs = len(step.observation.results) if step.observation else 0
            print(f"    step {step.step_id}: source={step.source} tool_calls={tc} observations={obs}")
        print()


if __name__ == "__main__":
    main()
