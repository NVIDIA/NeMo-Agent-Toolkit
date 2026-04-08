#!/usr/bin/env python3
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
"""Convert legacy `workflow_output.json` (IST) into `workflow_output_atif.json`.

Example:
    python packages/nvidia_nat_eval/scripts/convert_workflow_output_to_atif.py \
        --input ".tmp/nat/examples/advanced_agents/alert_triage_agent/output/offline_atif/workflow_output.json" \
        --output-dir ".tmp/nat/examples/advanced_agents/alert_triage_agent/output/offline_atif"
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from nat.data_models.intermediate_step import IntermediateStep
from nat.utils.atif_converter import IntermediateStepToATIFConverter


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _iter_items(payload: Any) -> list[dict[str, Any]]:
    """Normalize legacy workflow output payload into item dictionaries."""
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict) and isinstance(item.get("intermediate_steps"), list)]
    if isinstance(payload, dict) and isinstance(payload.get("intermediate_steps"), list):
        return [payload]
    raise ValueError("Unsupported input JSON shape. Expected item(s) with `intermediate_steps`.")


def _session_id_for_item(item_id: Any) -> str:
    item_type = type(item_id)
    return f"{item_type.__module__}.{item_type.__qualname__}:{item_id!r}"


def _convert_item(item: dict[str, Any],
                  *,
                  converter: IntermediateStepToATIFConverter,
                  item_index: int,
                  agent_name: str | None) -> dict[str, Any]:
    """Convert one legacy item with `intermediate_steps` to ATIF sample shape."""
    item_id = item.get("id", item_index)
    steps = [IntermediateStep.model_validate(raw_step) for raw_step in item["intermediate_steps"]]
    trajectory = converter.convert(
        steps=steps,
        session_id=_session_id_for_item(item_id),
        agent_name=agent_name,
    )

    subagent_trajectories: dict[str, Any] = {}
    if isinstance(trajectory.extra, dict):
        embedded = trajectory.extra.get("subagent_trajectories")
        if isinstance(embedded, dict):
            subagent_trajectories = embedded

    return {
        "item_id": item_id,
        "trajectory": trajectory.model_dump(mode="json"),
        "subagent_trajectories": subagent_trajectories,
        "expected_output_obj": item.get("answer"),
        "output_obj": item.get("generated_answer"),
        "metadata": {},
    }


def _resolve_output_path(input_path: Path, output_path: Path | None, output_dir: Path | None) -> Path:
    if output_path is not None:
        return output_path
    if output_dir is not None:
        return output_dir / "workflow_output_atif.json"
    return input_path.with_name("workflow_output_atif.json")


def main() -> None:
    """Parse arguments and run IST to ATIF conversion."""
    parser = argparse.ArgumentParser(
        description="Convert legacy workflow_output.json (IntermediateStep dump) to workflow_output_atif.json")
    parser.add_argument("--input", type=Path, required=True, help="Path to workflow_output.json")
    parser.add_argument("--output", type=Path, default=None, help="Output file path for workflow_output_atif.json")
    parser.add_argument("--output-dir", type=Path, default=None, help="Directory to write workflow_output_atif.json")
    parser.add_argument("--agent-name", default=None, help="Optional agent name override")
    parser.add_argument("--compact", action="store_true", help="Write compact JSON (no pretty indentation)")
    args = parser.parse_args()

    if args.output is not None and args.output_dir is not None:
        raise ValueError("Provide only one of `--output` or `--output-dir`.")

    payload = _load_json(args.input)
    items = _iter_items(payload)
    converter = IntermediateStepToATIFConverter()
    converted = [
        _convert_item(item, converter=converter, item_index=i, agent_name=args.agent_name)
        for i, item in enumerate(items)
    ]

    output_path = _resolve_output_path(args.input, args.output, args.output_dir)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_text = json.dumps(converted, ensure_ascii=True, indent=None if args.compact else 2)
    output_path.write_text(output_text + "\n", encoding="utf-8")
    print(f"Wrote {len(converted)} ATIF sample(s) to {output_path}")


if __name__ == "__main__":
    main()
