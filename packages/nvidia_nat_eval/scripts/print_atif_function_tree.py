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
"""Print a readable function ancestry tree from ATIF workflow output.

Example:
    python packages/nvidia_nat_eval/scripts/print_atif_function_tree.py \
        ".tmp/nat/examples/evaluation_and_profiling/simple_web_query_eval/atif/workflow_output_atif.json"
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class NodeStats:
    function_id: str
    function_name: str
    parent_id: str | None
    parent_name: str | None
    seen_in_step_ancestry: int = 0
    seen_in_tool_ancestry: int = 0


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _iter_trajectories(payload: Any) -> list[tuple[str, dict[str, Any]]]:
    """Normalize ATIF payload to (label, trajectory_dict)."""
    if isinstance(payload, list):
        out: list[tuple[str, dict[str, Any]]] = []
        for i, item in enumerate(payload):
            if not isinstance(item, dict):
                continue
            if isinstance(item.get("trajectory"), dict):
                label = f"item={item.get('item_id', i)}"
                out.append((label, item["trajectory"]))
            elif isinstance(item.get("steps"), list):
                out.append((f"trajectory={i}", item))
        return out

    if isinstance(payload, dict):
        if isinstance(payload.get("trajectory"), dict):
            return [(f"item={payload.get('item_id', '0')}", payload["trajectory"])]
        if isinstance(payload.get("steps"), list):
            return [("trajectory=0", payload)]

    raise ValueError("Unsupported ATIF JSON shape. Expected trajectory or eval-sample payload.")


def _add_ancestry(nodes: dict[str, NodeStats], fn: dict[str, Any], from_tool: bool) -> None:
    function_id = str(fn.get("function_id") or "")
    function_name = str(fn.get("function_name") or "")
    parent_id = fn.get("parent_id")
    parent_name = fn.get("parent_name")
    if not function_id or not function_name:
        return

    if function_id not in nodes:
        nodes[function_id] = NodeStats(
            function_id=function_id,
            function_name=function_name,
            parent_id=str(parent_id) if parent_id is not None else None,
            parent_name=str(parent_name) if parent_name is not None else None,
        )

    if from_tool:
        nodes[function_id].seen_in_tool_ancestry += 1
    else:
        nodes[function_id].seen_in_step_ancestry += 1


def _build_nodes(trajectory: dict[str, Any]) -> dict[str, NodeStats]:
    nodes: dict[str, NodeStats] = {}
    for step in trajectory.get("steps", []):
        extra = step.get("extra") or {}
        ancestry = extra.get("ancestry")
        if isinstance(ancestry, dict):
            _add_ancestry(nodes, ancestry.get("function_ancestry") or {}, from_tool=False)
        for tool_ancestry in extra.get("tool_ancestry") or []:
            if isinstance(tool_ancestry, dict):
                _add_ancestry(nodes, tool_ancestry.get("function_ancestry") or {}, from_tool=True)
    return nodes


def _print_tree(nodes: dict[str, NodeStats]) -> None:
    by_parent: dict[str, list[str]] = defaultdict(list)
    for function_id, node in nodes.items():
        parent = node.parent_id or "root"
        if parent == function_id:
            # Defensive guard against malformed self-parent links.
            parent = "root"
        by_parent[parent].append(function_id)

    for child_ids in by_parent.values():
        child_ids.sort(key=lambda fid: nodes[fid].function_name)

    roots = [fid for fid, node in nodes.items() if (node.parent_id in (None, "", "root")) and fid != "root"]
    roots.sort(key=lambda fid: nodes[fid].function_name)

    def rec(function_id: str, prefix: str, is_last: bool, visited: set[str]) -> None:
        if function_id in visited:
            branch = "└─ " if is_last else "├─ "
            print(f"{prefix}{branch}<cycle> [{function_id}]")
            return
        visited = set(visited)
        visited.add(function_id)

        node = nodes[function_id]
        branch = "└─ " if is_last else "├─ "
        counts = []
        if node.seen_in_step_ancestry:
            counts.append(f"steps={node.seen_in_step_ancestry}")
        if node.seen_in_tool_ancestry:
            counts.append(f"tools={node.seen_in_tool_ancestry}")
        counts_str = f" ({', '.join(counts)})" if counts else ""
        print(f"{prefix}{branch}{node.function_name} [{node.function_id}]{counts_str}")

        children = by_parent.get(function_id, [])
        child_prefix = prefix + ("   " if is_last else "│  ")
        for i, child_id in enumerate(children):
            rec(child_id, child_prefix, i == len(children) - 1, visited)

    print("root")
    if not roots and "root" in nodes:
        roots = ["root"]
    for i, root_id in enumerate(roots):
        rec(root_id, "", i == len(roots) - 1, set())


def main() -> None:
    parser = argparse.ArgumentParser(description="Print ATIF function ancestry tree from workflow_output_atif.json")
    parser.add_argument("input_json", type=Path, help="Path to ATIF workflow output JSON")
    args = parser.parse_args()

    payload = _load_json(args.input_json)
    trajectories = _iter_trajectories(payload)

    for idx, (label, trajectory) in enumerate(trajectories):
        if idx > 0:
            print()
        session_id = trajectory.get("session_id", "unknown-session")
        print(f"=== {label} | mode=atif | session_id={session_id} ===")
        nodes = _build_nodes(trajectory)
        if not nodes:
            print("No ancestry metadata found in step.extra.")
            continue
        _print_tree(nodes)


if __name__ == "__main__":
    main()
