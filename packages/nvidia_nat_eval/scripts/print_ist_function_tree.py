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
"""Print a readable function ancestry tree from legacy IST workflow output.

Example:
    python packages/nvidia_nat_eval/scripts/print_ist_function_tree.py \
        ".tmp/nat/examples/evaluation_and_profiling/simple_web_query_eval/atif/workflow_output.json"
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


def _iter_items(payload: Any) -> list[tuple[str, dict[str, Any]]]:
    """Normalize legacy payload to (label, item_dict)."""
    if isinstance(payload, list):
        out: list[tuple[str, dict[str, Any]]] = []
        for i, item in enumerate(payload):
            if isinstance(item, dict) and isinstance(item.get("intermediate_steps"), list):
                out.append((f"item={item.get('id', i)}", item))
        return out

    if isinstance(payload, dict):
        if isinstance(payload.get("intermediate_steps"), list):
            return [(f"item={payload.get('id', '0')}", payload)]

    raise ValueError("Unsupported legacy JSON shape. Expected item(s) with intermediate_steps.")


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


def _build_nodes(item: dict[str, Any]) -> dict[str, NodeStats]:
    nodes: dict[str, NodeStats] = {}
    for step in item.get("intermediate_steps", []):
        fn = step.get("function_ancestry")
        if not isinstance(fn, dict):
            continue
        event_type = ((step.get("payload") or {}).get("event_type") or "")
        _add_ancestry(nodes, fn, from_tool=("TOOL" in str(event_type)))
    return nodes


def _extract_nested_tool_chain(step: dict[str, Any], tool_name: str) -> list[str]:
    """Infer nested tool calls from optional NAT function end events."""
    payload = step.get("payload") or {}
    metadata = payload.get("metadata") or {}
    nat_events = metadata.get("nat_events") or payload.get("nat_events") or []

    names: list[str] = []
    for event in nat_events:
        if not isinstance(event, dict):
            continue
        if event.get("type") != "FUNCTION_END":
            continue
        name = event.get("name")
        if isinstance(name, str) and name and name != "<workflow>":
            names.append(name)

    ordered = list(dict.fromkeys(reversed(names)))
    if ordered and ordered[0] == tool_name:
        ordered = ordered[1:]
    return ordered


def _build_execution_graph(item: dict[str, Any]) -> tuple[dict[str, set[str]], dict[str, int]]:
    """
    Build an inferred execution graph:
    root -> <workflow> -> <llm:model> -> tool -> nested tools.
    """
    edges: dict[str, set[str]] = defaultdict(set)
    seen_counts: dict[str, int] = defaultdict(int)

    workflow_node = "<workflow>"
    edges["root"].add(workflow_node)
    seen_counts[workflow_node] += 1

    for step in item.get("intermediate_steps", []):
        if not isinstance(step, dict):
            continue
        payload = step.get("payload") or {}
        event_type = str(payload.get("event_type") or "")
        parent_node = workflow_node

        if event_type == "LLM_END":
            model_name = payload.get("name")
            if isinstance(model_name, str) and model_name:
                llm_node = f"<llm:{model_name}>"
                edges[workflow_node].add(llm_node)
                seen_counts[llm_node] += 1
            continue

        if event_type != "TOOL_END":
            continue

        ancestry = step.get("function_ancestry")
        if isinstance(ancestry, dict):
            # Attach tools to the workflow represented by ancestry.
            parent_name = ancestry.get("function_name")
            if isinstance(parent_name, str) and parent_name:
                parent_node = parent_name

        tool_name = payload.get("name")
        if not isinstance(tool_name, str) or not tool_name:
            continue

        chain = [tool_name, *_extract_nested_tool_chain(step, tool_name)]
        prev = parent_node
        for node in chain:
            edges[prev].add(node)
            seen_counts[node] += 1
            prev = node

    return edges, seen_counts


def _build_execution_chains(item: dict[str, Any]) -> list[list[str]]:
    """Build per-occurrence execution chains for explicit sequence visualization."""
    chains: list[list[str]] = []
    workflow_node = "<workflow>"

    for step in item.get("intermediate_steps", []):
        if not isinstance(step, dict):
            continue
        payload = step.get("payload") or {}
        event_type = str(payload.get("event_type") or "")

        if event_type == "LLM_END":
            model_name = payload.get("name")
            if isinstance(model_name, str) and model_name:
                chains.append([workflow_node, f"<llm:{model_name}>"])
            continue

        if event_type != "TOOL_END":
            continue

        tool_name = payload.get("name")
        if not isinstance(tool_name, str) or not tool_name:
            continue
        chain = [workflow_node, tool_name]
        chain.extend(_extract_nested_tool_chain(step, tool_name))
        chains.append(chain)

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

    roots = sorted(edges.get("root", set()))
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
    """Parse CLI args, load legacy IST JSON, and print function ancestry trees."""
    parser = argparse.ArgumentParser(description="Print legacy IST function ancestry tree from workflow_output.json")
    parser.add_argument("input_json", type=Path, help="Path to legacy workflow output JSON")
    parser.add_argument(
        "--view",
        choices=["ancestry", "execution", "execution_sequence"],
        default="ancestry",
        help=("Tree view type. 'ancestry' uses recorded ancestry metadata. "
              "'execution' shows an aggregated runtime chain graph. "
              "'execution_sequence' lists each runtime occurrence as its own branch."),
    )
    parser.add_argument(
        "--item-id",
        help="Only print a specific item id (for example: 3).",
    )
    args = parser.parse_args()

    payload = _load_json(args.input_json)
    items = _iter_items(payload)

    if args.item_id is not None:
        items = [(label, item) for (label, item) in items if _label_id(label) == str(args.item_id)]
        if not items:
            print(f"No item found for item_id={args.item_id}")
            return

    for idx, (label, item) in enumerate(items):
        if idx > 0:
            print()
        session_id = item.get("session_id", "unknown-session")
        print(f"=== {label} | mode=legacy | session_id={session_id} ===")
        if args.view == "execution":
            edges, seen_counts = _build_execution_graph(item)
            if not edges:
                print("No execution metadata found in intermediate_steps.")
                continue
            _print_execution_tree(edges, seen_counts)
        elif args.view == "execution_sequence":
            chains = _build_execution_chains(item)
            if not chains:
                print("No execution metadata found in intermediate_steps.")
                continue
            _print_execution_sequence_tree(chains)
        else:
            nodes = _build_nodes(item)
            if not nodes:
                print("No function_ancestry metadata found in intermediate_steps.")
                continue
            _print_tree(nodes)


if __name__ == "__main__":
    main()
