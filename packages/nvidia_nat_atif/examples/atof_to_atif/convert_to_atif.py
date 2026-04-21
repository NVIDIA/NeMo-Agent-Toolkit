# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Convert ATOF v0.1 JSONL examples to ATIF trajectories.

Reads each example JSONL file produced by ``generate_examples.py`` (the three
three-tier demonstration scenarios: EXMP-01 tier-2 basic, EXMP-02 tier-2 with
error recovery, EXMP-03 tier-3 schema-annotated) and writes the resulting ATIF
trajectory as formatted JSON.

Uses ``nat.atof.scripts.atof_to_atif_converter.convert_file`` — the v0.1
converter dispatches on ``(kind, scope_type)`` and reads scope-type-specific
fields from the ``profile`` sub-object (``profile.model_name``,
``profile.tool_call_id``). ``StreamHeader`` events are skipped by the
converter's main dispatch loop (no schema-registry pre-pass needed in v0.1;
the optional schema layer is consumer-side opt-in).

Usage:
    python convert_to_atif.py [--input-dir DIR] [--output-dir DIR]
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
    parser = argparse.ArgumentParser(description="Convert ATOF v0.2 JSONL to ATIF JSON")
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
