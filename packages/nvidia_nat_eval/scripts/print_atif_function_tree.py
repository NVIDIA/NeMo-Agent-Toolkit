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
    """Stats for a function node in the ancestry tree."""

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


def _label_id(label: str) -> str:
    """Extract the ID portion from a normalized label like item=4."""
    return label.split("=", 1)[1] if "=" in label else label


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


def _normalize_path_nodes(raw_path: Any) -> list[dict[str, Any]]:
    """Return validated path nodes from a raw path value."""
    if not isinstance(raw_path, list):
        return []
    return [n for n in raw_path if isinstance(n, dict)]


def _iter_step_paths(step: dict[str, Any]) -> tuple[list[dict[str, Any]], list[list[dict[str, Any]]]]:
    """Extract canonical step/tool ancestry paths from `step.extra`."""
    extra = step.get("extra") or {}
    if not isinstance(extra, dict):
        return [], []

    step_path = _normalize_path_nodes(extra.get("step_ancestry_path"))

    tool_paths: list[list[dict[str, Any]]] = []
    raw_tool_paths = extra.get("tool_ancestry_paths")
    if isinstance(raw_tool_paths, list):
        for raw_path in raw_tool_paths:
            path_nodes = _normalize_path_nodes(raw_path)
            if path_nodes:
                tool_paths.append(path_nodes)

    return step_path, tool_paths


def _add_path(nodes: dict[str, NodeStats], path: list[dict[str, Any]], from_tool: bool) -> None:
    """Add all nodes from one ancestry path to aggregated stats."""
    for idx, fn in enumerate(path):
        if idx > 0:
            # Enforce parent relationship from path adjacency when possible.
            prev = path[idx - 1]
            fn = dict(fn)
            fn.setdefault("parent_id", prev.get("function_id"))
            fn.setdefault("parent_name", prev.get("function_name"))
        _add_ancestry(nodes, fn, from_tool=from_tool)


def _build_nodes(trajectory: dict[str, Any]) -> dict[str, NodeStats]:
    """Build node stats from convenience path fields in `step.extra`."""
    nodes: dict[str, NodeStats] = {}
    for step in trajectory.get("steps", []):
        if not isinstance(step, dict):
            continue
        step_path, tool_paths = _iter_step_paths(step)
        if step_path:
            _add_path(nodes, step_path, from_tool=False)
        for tool_path in tool_paths:
            _add_path(nodes, tool_path, from_tool=True)
    return nodes


def _build_nodes_from_required_ancestry(trajectory: dict[str, Any]) -> dict[str, NodeStats]:
    """Build node stats from required ancestry fields in `step.extra`.

    This uses:
    - `extra.ancestry.function_ancestry`
    - `extra.tool_ancestry[].function_ancestry`
    """
    nodes: dict[str, NodeStats] = {}
    for step in trajectory.get("steps", []):
        if not isinstance(step, dict):
            continue
        extra = step.get("extra") or {}
        if not isinstance(extra, dict):
            continue

        ancestry = extra.get("ancestry")
        if isinstance(ancestry, dict):
            fn = ancestry.get("function_ancestry")
            if isinstance(fn, dict):
                _add_ancestry(nodes, fn, from_tool=False)

        for tool_ancestry in extra.get("tool_ancestry") or []:
            if not isinstance(tool_ancestry, dict):
                continue
            fn = tool_ancestry.get("function_ancestry")
            if isinstance(fn, dict):
                _add_ancestry(nodes, fn, from_tool=True)

    return nodes


def _path_to_labels(path: list[dict[str, Any]]) -> list[str]:
    """Convert path nodes to stable display labels."""
    labels: list[str] = []
    for node in path:
        function_id = str(node.get("function_id") or "")
        function_name = str(node.get("function_name") or "")
        if not function_id or not function_name:
            continue
        if function_id == "root":
            # Skip explicit root node; the printer already has a synthetic root.
            continue
        labels.append(f"{function_name} [{function_id}]")
    return labels


def _label_function_name(label: str) -> str:
    """Extract function name from a display label."""
    if " [" in label and label.endswith("]"):
        return label.rsplit(" [", 1)[0]
    return label


def _extract_tool_call_names(step: dict[str, Any]) -> list[str]:
    """Extract tool call names from a step in order."""
    names: list[str] = []
    for tool_call in step.get("tool_calls") or []:
        if not isinstance(tool_call, dict):
            continue
        name = tool_call.get("function_name")
        if isinstance(name, str) and name:
            names.append(name)
    return names


def _build_step_tool_chain(step: dict[str, Any], tool_idx: int, tool_name: str) -> list[str]:
    """Build one execution chain for a tool call from path metadata + tool call data."""
    step_path, tool_paths = _iter_step_paths(step)
    chain = _path_to_labels(step_path)

    model_name = step.get("model_name")
    if isinstance(model_name, str) and model_name:
        llm_label = f"<llm:{model_name}>"
        if not chain or chain[-1] != llm_label:
            chain.append(llm_label)

    # If tool path has deeper lineage than step path, append only the delta.
    if tool_idx < len(tool_paths):
        tool_labels = _path_to_labels(tool_paths[tool_idx])
        prefix_len = len(_path_to_labels(step_path))
        for label in tool_labels[prefix_len:]:
            if not chain or chain[-1] != label:
                chain.append(label)

    # Ensure the explicit tool call is represented even when tool path is shallow.
    has_tool_name = any(_label_function_name(label) == tool_name for label in chain)
    if not has_tool_name:
        chain.append(tool_name)

    return chain


def _build_execution_graph(trajectory: dict[str, Any]) -> tuple[dict[str, set[str]], dict[str, int]]:
    """Build an aggregated execution graph from canonical path lists."""
    edges: dict[str, set[str]] = defaultdict(set)
    seen_counts: dict[str, int] = defaultdict(int)
    root_node = "__root__"

    for step in trajectory.get("steps", []):
        if not isinstance(step, dict):
            continue
        step_path, _ = _iter_step_paths(step)

        step_labels = _path_to_labels(step_path)
        model_name = step.get("model_name")
        if isinstance(model_name, str) and model_name:
            llm_label = f"<llm:{model_name}>"
            if not step_labels or step_labels[-1] != llm_label:
                step_labels = [*step_labels, llm_label]

        if step_labels:
            edges[root_node].add(step_labels[0])
            for label in step_labels:
                seen_counts[label] += 1
            for idx in range(1, len(step_labels)):
                edges[step_labels[idx - 1]].add(step_labels[idx])

        for tool_idx, tool_name in enumerate(_extract_tool_call_names(step)):
            tool_chain = _build_step_tool_chain(step, tool_idx, tool_name)
            if not tool_chain:
                continue
            edges[root_node].add(tool_chain[0])
            for label in tool_chain:
                seen_counts[label] += 1
            for idx in range(1, len(tool_chain)):
                edges[tool_chain[idx - 1]].add(tool_chain[idx])

    return edges, seen_counts


def _build_execution_chains(trajectory: dict[str, Any]) -> list[list[str]]:
    """Build per-occurrence execution chains from canonical path lists."""
    chains: list[list[str]] = []

    for step in trajectory.get("steps", []):
        if not isinstance(step, dict):
            continue
        step_path, tool_paths = _iter_step_paths(step)
        if step_path:
            labels = _path_to_labels(step_path)
            model_name = step.get("model_name")
            if isinstance(model_name, str) and model_name:
                llm_label = f"<llm:{model_name}>"
                if not labels or labels[-1] != llm_label:
                    labels.append(llm_label)
            if labels:
                chains.append(labels)

        tool_names = _extract_tool_call_names(step)
        if tool_names:
            for idx, tool_name in enumerate(tool_names):
                labels = _build_step_tool_chain(step, idx, tool_name)
                if labels:
                    chains.append(labels)
        else:
            for tool_path in tool_paths:
                labels = _path_to_labels(tool_path)
                if labels:
                    chains.append(labels)

    return chains


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

    roots = [
        fid for fid, node in nodes.items()
        if (fid != "root" and (node.parent_id in (None, "", "root") or node.parent_id not in nodes))
    ]
    roots.sort(key=lambda fid: nodes[fid].function_name)

    covered: set[str] = set()

    def rec(function_id: str, prefix: str, is_last: bool, visited: set[str]) -> None:
        if function_id in visited:
            branch = "└─ " if is_last else "├─ "
            print(f"{prefix}{branch}<cycle> [{function_id}]")
            return
        visited = set(visited)
        visited.add(function_id)
        covered.add(function_id)

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

    # Ensure disconnected/cyclic components are still surfaced as top-level entries.
    remaining_roots = sorted((fid for fid in nodes if fid not in covered), key=lambda fid: nodes[fid].function_name)
    for i, root_id in enumerate(remaining_roots):
        rec(root_id, "", i == len(remaining_roots) - 1, set())


def _print_execution_tree(edges: dict[str, set[str]], seen_counts: dict[str, int]) -> None:
    """Print the inferred execution graph as a readable tree."""
    root_key = "__root__"
    print("root")

    def rec(node: str, prefix: str, is_last: bool, visited: set[str]) -> None:
        branch = "└─ " if is_last else "├─ "
        if node in visited:
            print(f"{prefix}{branch}<cycle> [{node}]")
            return

        visited = set(visited)
        visited.add(node)
        count = seen_counts.get(node, 0)
        count_suffix = f" (seen={count})" if count else ""
        print(f"{prefix}{branch}{node}{count_suffix}")

        children = sorted(edges.get(node, set()))
        child_prefix = prefix + ("   " if is_last else "│  ")
        for idx, child in enumerate(children):
            rec(child, child_prefix, idx == len(children) - 1, visited)

    roots = sorted(edges.get(root_key, set()))
    for idx, root_node in enumerate(roots):
        rec(root_node, "", idx == len(roots) - 1, set())


def _print_execution_sequence_tree(chains: list[list[str]]) -> None:
    """Print each execution occurrence as an explicit branch."""
    print("root")
    if not chains:
        return

    for i, chain in enumerate(chains, start=1):
        run_branch = "└─ " if i == len(chains) else "├─ "
        print(f"{run_branch}run_{i}")
        prefix = "   " if i == len(chains) else "│  "
        for j, node in enumerate(chain):
            node_branch = "└─ " if j == len(chain) - 1 else "├─ "
            print(f"{prefix}{node_branch}{node}")
            prefix += "   " if j == len(chain) - 1 else "│  "


def main() -> None:
    """Parse the input JSON and print the ATIF function ancestry tree."""
    parser = argparse.ArgumentParser(description="Print ATIF function ancestry tree from workflow_output_atif.json")
    parser.add_argument("input_json", type=Path, help="Path to ATIF workflow output JSON")
    parser.add_argument(
        "--view",
        choices=[
            "ancestry", "ancestry_paths", "ancestry_required", "ancestry_compare", "execution", "execution_sequence"
        ],
        default="ancestry",
        help=("Tree view type. 'ancestry'/'ancestry_compare' prints both convenience path-based and required "
              "ancestry trees for A/B comparison. "
              "'ancestry_paths' uses convenience path-based ancestry metadata only. "
              "'ancestry_required' uses required ancestry fields (`ancestry`, `tool_ancestry`) only. "
              "'execution' shows an aggregated path graph. "
              "'execution_sequence' lists each path occurrence as its own branch."),
    )
    parser.add_argument(
        "--item-id",
        help="Only print a specific item_id (for example: 3).",
    )
    args = parser.parse_args()

    payload = _load_json(args.input_json)
    trajectories = _iter_trajectories(payload)

    if args.item_id is not None:
        trajectories = [(label, t) for (label, t) in trajectories if _label_id(label) == str(args.item_id)]
        if not trajectories:
            print(f"No trajectory found for item_id={args.item_id}")
            return

    for idx, (label, trajectory) in enumerate(trajectories):
        if idx > 0:
            print()
        session_id = trajectory.get("session_id", "unknown-session")
        print(f"=== {label} | mode=atif | session_id={session_id} ===")
        if args.view == "execution":
            edges, seen_counts = _build_execution_graph(trajectory)
            if not edges:
                print("No path metadata found in trajectory steps.")
                continue
            _print_execution_tree(edges, seen_counts)
        elif args.view == "execution_sequence":
            chains = _build_execution_chains(trajectory)
            if not chains:
                print("No path metadata found in trajectory steps.")
                continue
            _print_execution_sequence_tree(chains)
        else:
            if args.view in ("ancestry", "ancestry_compare", "ancestry_paths"):
                path_nodes = _build_nodes(trajectory)
            else:
                path_nodes = {}
            if args.view in ("ancestry", "ancestry_compare", "ancestry_required"):
                required_nodes = _build_nodes_from_required_ancestry(trajectory)
            else:
                required_nodes = {}

            if args.view in ("ancestry_paths", ):
                if not path_nodes:
                    print("No convenience path ancestry metadata found in step.extra.")
                    continue
                print("--- convenience_paths ---")
                _print_tree(path_nodes)
                continue

            if args.view in ("ancestry_required", ):
                if not required_nodes:
                    print("No required ancestry metadata (`ancestry`, `tool_ancestry`) found in step.extra.")
                    continue
                print("--- required_ancestry ---")
                _print_tree(required_nodes)
                continue

            # Compare/default mode: print both trees for easy A/B visual diff.
            if not path_nodes and not required_nodes:
                print("No ancestry metadata found in step.extra.")
                continue
            print("--- convenience_paths ---")
            if path_nodes:
                _print_tree(path_nodes)
            else:
                print("(missing)")
            print()
            print("--- required_ancestry ---")
            if required_nodes:
                _print_tree(required_nodes)
            else:
                print("(missing)")


if __name__ == "__main__":
    main()
