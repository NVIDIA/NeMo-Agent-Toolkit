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
import sys
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
    """Normalize ATIF payload to (label, trajectory_dict).

    Supported input shapes include:
    - eval output wrappers: [{"item_id": ..., "trajectory": {...}}, ...]
    - single eval wrapper: {"item_id": ..., "trajectory": {...}}
    - raw trajectory object: {"schema_version": "...", "steps": [...]}
    - list of raw trajectory objects
    - nested containers that include any of the above
    """

    def _is_trajectory(obj: Any) -> bool:
        return isinstance(obj, dict) and isinstance(obj.get("steps"), list)

    def _collect(item: Any, label_prefix: str) -> list[tuple[str, dict[str, Any]]]:
        out: list[tuple[str, dict[str, Any]]] = []

        if isinstance(item, dict):
            # Preferred eval wrapper form.
            if isinstance(item.get("trajectory"), dict) and _is_trajectory(item.get("trajectory")):
                label = f"item={item.get('item_id', label_prefix)}"
                out.append((label, item["trajectory"]))
                return out

            # Raw trajectory object.
            if _is_trajectory(item):
                out.append((f"trajectory={label_prefix}", item))
                return out

            # Recurse through nested mappings.
            for key, value in item.items():
                out.extend(_collect(value, f"{label_prefix}.{key}"))
            return out

        if isinstance(item, list):
            for i, value in enumerate(item):
                out.extend(_collect(value, f"{label_prefix}[{i}]"))
            return out

        return out

    collected = _collect(payload, "root")
    if collected:
        return collected

    # Backward-compatible fallback path.
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

    raise ValueError("Unsupported ATIF JSON shape. No trajectory object with a 'steps' array was found.")


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


def _build_nodes(trajectory: dict[str, Any]) -> dict[str, NodeStats]:
    """Build node stats from required ancestry fields in `step.extra`."""
    return _build_nodes_from_required_ancestry(trajectory)


def _build_nodes_from_required_ancestry(trajectory: dict[str, Any]) -> dict[str, NodeStats]:
    """Build node stats from required ancestry fields in `step.extra`.

    This uses:
    - `extra.ancestry`
    - `extra.tool_ancestry[]`
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
            _add_ancestry(nodes, ancestry, from_tool=False)

        for tool_ancestry in extra.get("tool_ancestry") or []:
            if not isinstance(tool_ancestry, dict):
                continue
            _add_ancestry(nodes, tool_ancestry, from_tool=True)

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


def _extract_step_function_node(step: dict[str, Any]) -> dict[str, Any] | None:
    """Extract step-level function ancestry node."""
    extra = step.get("extra") or {}
    if not isinstance(extra, dict):
        return None
    ancestry = extra.get("ancestry")
    if not isinstance(ancestry, dict):
        return None
    return ancestry


def _extract_tool_function_nodes(step: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract tool-level function ancestry nodes aligned with `tool_calls`."""
    extra = step.get("extra") or {}
    if not isinstance(extra, dict):
        return []
    out: list[dict[str, Any]] = []
    for item in extra.get("tool_ancestry") or []:
        if not isinstance(item, dict):
            continue
        out.append(item)
    return out


def _build_step_tool_chain(step: dict[str, Any], tool_idx: int, tool_name: str) -> list[str]:
    """Build one execution chain for a tool call from required ancestry fields."""
    chain: list[str] = []
    step_node = _extract_step_function_node(step)
    if step_node is not None:
        chain.extend(_path_to_labels([step_node]))

    model_name = step.get("model_name")
    if isinstance(model_name, str) and model_name:
        llm_label = f"<llm:{model_name}>"
        if not chain or chain[-1] != llm_label:
            chain.append(llm_label)

    tool_nodes = _extract_tool_function_nodes(step)
    if tool_idx < len(tool_nodes):
        tool_labels = _path_to_labels([tool_nodes[tool_idx]])
        for label in tool_labels:
            if not chain or chain[-1] != label:
                chain.append(label)

    # Ensure the explicit tool call is represented even when tool path is shallow.
    has_tool_name = any(_label_function_name(label) == tool_name for label in chain)
    if not has_tool_name:
        chain.append(tool_name)

    return chain


def _build_execution_graph(trajectory: dict[str, Any]) -> tuple[dict[str, set[str]], dict[str, int]]:
    """Build an aggregated execution graph from required ancestry fields."""
    edges: dict[str, set[str]] = defaultdict(set)
    seen_counts: dict[str, int] = defaultdict(int)
    root_node = "__root__"

    for step in trajectory.get("steps", []):
        if not isinstance(step, dict):
            continue
        step_node = _extract_step_function_node(step)
        step_labels = _path_to_labels([step_node] if step_node is not None else [])
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
    """Build per-occurrence execution chains from required ancestry fields."""
    chains: list[list[str]] = []

    for step in trajectory.get("steps", []):
        if not isinstance(step, dict):
            continue
        step_node = _extract_step_function_node(step)
        labels = _path_to_labels([step_node] if step_node is not None else [])
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

    return chains


def _print_tree(nodes: dict[str, NodeStats]) -> None:
    root_stats = nodes.get("root")
    by_parent: dict[str, list[str]] = defaultdict(list)
    for function_id, node in nodes.items():
        if function_id == "root":
            # The printer already emits a synthetic root header; avoid treating
            # the explicit root node as a child, which creates duplicate subtrees.
            continue
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

    if root_stats is not None:
        counts = []
        if root_stats.seen_in_step_ancestry:
            counts.append(f"steps={root_stats.seen_in_step_ancestry}")
        if root_stats.seen_in_tool_ancestry:
            counts.append(f"tools={root_stats.seen_in_tool_ancestry}")
        counts_str = f" ({', '.join(counts)})" if counts else ""
        print(f"root{counts_str}")
    else:
        print("root")

    for i, root_id in enumerate(roots):
        rec(root_id, "", i == len(roots) - 1, set())

    # Ensure disconnected/cyclic components are still surfaced as top-level entries.
    remaining_roots = sorted((fid for fid in nodes if fid != "root" and fid not in covered),
                             key=lambda fid: nodes[fid].function_name)
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


def _step_summary(trajectory: dict[str, Any]) -> tuple[int, int, int, int]:
    """Return total steps, user steps, agent steps, and total tool calls."""
    steps = trajectory.get("steps", [])
    if not isinstance(steps, list):
        return 0, 0, 0, 0

    total_steps = 0
    user_steps = 0
    agent_steps = 0
    total_tool_calls = 0

    for step in steps:
        if not isinstance(step, dict):
            continue
        total_steps += 1
        if step.get("source") == "user":
            user_steps += 1
        elif step.get("source") == "agent":
            agent_steps += 1
        tool_calls = step.get("tool_calls")
        if isinstance(tool_calls, list):
            total_tool_calls += len(tool_calls)

    return total_steps, user_steps, agent_steps, total_tool_calls


def _print_step_breakdown(trajectory: dict[str, Any]) -> None:
    """Print a compact per-step breakdown for quick count reconciliation."""
    steps = trajectory.get("steps", [])
    if not isinstance(steps, list):
        print("steps:")
        print("  (none)")
        return

    print("steps:")
    for idx, step in enumerate(steps, start=1):
        if not isinstance(step, dict):
            continue
        source = step.get("source", "?")
        extra = step.get("extra") or {}
        ancestry = extra.get("ancestry") if isinstance(extra, dict) else None
        fn_name = ancestry.get("function_name") if isinstance(ancestry, dict) else "?"
        fn_id = ancestry.get("function_id") if isinstance(ancestry, dict) else "?"
        tool_calls = step.get("tool_calls")
        tool_count = len(tool_calls) if isinstance(tool_calls, list) else 0
        print(f"  {idx:>2}. source={source:<5} ancestry={fn_name} [{fn_id}] tool_calls={tool_count}")


def _validate_trajectory_contract(trajectory: dict[str, Any]) -> list[str]:
    """Validate ATIF lineage/invocation contract invariants for one trajectory."""
    issues: list[str] = []
    steps = trajectory.get("steps", [])
    if not isinstance(steps, list):
        return ["steps is not a list"]

    known_function_ids: set[str] = {"root"}
    lineage_nodes: list[tuple[int, str, str | None]] = []

    for step_idx, step in enumerate(steps, start=1):
        if not isinstance(step, dict):
            issues.append(f"step {step_idx}: step is not an object")
            continue

        extra = step.get("extra") or {}
        if not isinstance(extra, dict):
            issues.append(f"step {step_idx}: extra is not an object")
            continue

        # Step-level ancestry node collection for parent-chain validation.
        ancestry = extra.get("ancestry")
        if isinstance(ancestry, dict):
            function_id = str(ancestry.get("function_id") or "")
            parent_id = str(ancestry.get("parent_id")) if ancestry.get("parent_id") is not None else None
            if function_id:
                known_function_ids.add(function_id)
                lineage_nodes.append((step_idx, function_id, parent_id))

        tool_calls = step.get("tool_calls") or []
        if not isinstance(tool_calls, list):
            issues.append(f"step {step_idx}: tool_calls is not a list")
            tool_calls = []

        tool_ancestry = extra.get("tool_ancestry") or []
        if not isinstance(tool_ancestry, list):
            issues.append(f"step {step_idx}: tool_ancestry is not a list")
            tool_ancestry = []

        tool_invocations_raw = extra.get("tool_invocations")
        tool_invocations = tool_invocations_raw if isinstance(tool_invocations_raw, list) else None
        if tool_invocations_raw is not None and tool_invocations is None:
            issues.append(f"step {step_idx}: tool_invocations is not a list")

        # Invariant: aligned arrays.
        if tool_calls and len(tool_ancestry) != len(tool_calls):
            issues.append(
                f"step {step_idx}: len(tool_ancestry)={len(tool_ancestry)} != len(tool_calls)={len(tool_calls)}")
        if tool_invocations is not None and len(tool_invocations) != len(tool_calls):
            issues.append(
                f"step {step_idx}: len(tool_invocations)={len(tool_invocations)} != len(tool_calls)={len(tool_calls)}")

        # Invariant: unique call IDs per step and observation linkage.
        obs_results = (step.get("observation") or {}).get("results") or []
        obs_ids = {r.get("source_call_id") for r in obs_results if isinstance(r, dict) and r.get("source_call_id")}
        seen_call_ids: set[str] = set()
        for i, tool_call in enumerate(tool_calls):
            if not isinstance(tool_call, dict):
                issues.append(f"step {step_idx}: tool_calls[{i}] is not an object")
                continue
            call_id = str(tool_call.get("tool_call_id") or "")
            if not call_id:
                issues.append(f"step {step_idx}: tool_calls[{i}] missing tool_call_id")
                continue
            if call_id in seen_call_ids:
                issues.append(f"step {step_idx}: duplicate tool_call_id {call_id}")
            seen_call_ids.add(call_id)
            if call_id not in obs_ids:
                issues.append(f"step {step_idx}: missing observation source_call_id for {call_id}")

            if i < len(tool_ancestry):
                ta = tool_ancestry[i]
                if not isinstance(ta, dict):
                    issues.append(f"step {step_idx}: tool_ancestry[{i}] missing ancestry node")
                else:
                    function_id = str(ta.get("function_id") or "")
                    parent_id = str(ta.get("parent_id")) if ta.get("parent_id") is not None else None
                    if not function_id:
                        issues.append(f"step {step_idx}: tool_ancestry[{i}] missing function_id")
                    else:
                        known_function_ids.add(function_id)
                        lineage_nodes.append((step_idx, function_id, parent_id))

            if tool_invocations is not None and i < len(tool_invocations):
                inv = tool_invocations[i] if isinstance(tool_invocations[i], dict) else {}
                start_ts = inv.get("start_timestamp")
                end_ts = inv.get("end_timestamp")
                if (start_ts is None) ^ (end_ts is None):
                    issues.append(f"step {step_idx}: tool_invocations[{i}] has partial timestamps")

        # Invariant: step-level invocation timestamp pairing.
        invocation = extra.get("invocation")
        if isinstance(invocation, dict):
            start_ts = invocation.get("start_timestamp")
            end_ts = invocation.get("end_timestamp")
            if (start_ts is None) ^ (end_ts is None):
                issues.append(f"step {step_idx}: invocation has partial timestamps")

    # Invariant: parent chain references resolve to known nodes or root.
    for step_idx, function_id, parent_id in lineage_nodes:
        if parent_id in (None, "", "root"):
            continue
        if parent_id not in known_function_ids:
            issues.append(f"step {step_idx}: parent_id {parent_id} for {function_id} not found in trajectory lineage")

    return issues


def main() -> None:
    """Parse the input JSON and print the ATIF function ancestry tree."""
    parser = argparse.ArgumentParser(
        description="Print ATIF function ancestry tree from any JSON payload containing trajectory objects.")
    parser.add_argument("input_json", type=Path, help="Path to ATIF workflow output JSON")
    parser.add_argument(
        "--view",
        choices=["ancestry", "execution", "execution_sequence"],
        default="ancestry",
        help=("Tree view type. 'ancestry' uses required ancestry fields (`ancestry`, `tool_ancestry`). "
              "'execution' shows an aggregated execution graph. "
              "'execution_sequence' lists each execution occurrence as its own branch."),
    )
    parser.add_argument(
        "--item-id",
        help="Only print a specific item_id (for example: 3).",
    )
    parser.add_argument(
        "--show-steps",
        action="store_true",
        help="Print per-step source/ancestry/tool-call breakdown before the tree.",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help=("Validate ATIF lineage/invocation contract invariants and return non-zero exit code "
              "when violations are found."),
    )
    args = parser.parse_args()

    payload = _load_json(args.input_json)
    trajectories = _iter_trajectories(payload)

    if args.item_id is not None:
        trajectories = [(label, t) for (label, t) in trajectories if _label_id(label) == str(args.item_id)]
        if not trajectories:
            print(f"No trajectory found for item_id={args.item_id}")
            return

    had_validation_errors = False
    for idx, (label, trajectory) in enumerate(trajectories):
        if idx > 0:
            print()
        session_id = trajectory.get("session_id", "unknown-session")
        total_steps, user_steps, agent_steps, total_tool_calls = _step_summary(trajectory)
        print(f"=== {label} | mode=atif | session_id={session_id} ===")
        print("summary: "
              f"steps={total_steps} (user={user_steps}, agent={agent_steps}), "
              f"tool_calls={total_tool_calls}")
        if args.show_steps:
            _print_step_breakdown(trajectory)
        if args.view == "execution":
            edges, seen_counts = _build_execution_graph(trajectory)
            if not edges:
                print("No ancestry metadata found in trajectory steps.")
                continue
            _print_execution_tree(edges, seen_counts)
        elif args.view == "execution_sequence":
            chains = _build_execution_chains(trajectory)
            if not chains:
                print("No ancestry metadata found in trajectory steps.")
                continue
            _print_execution_sequence_tree(chains)
        else:
            required_nodes = _build_nodes_from_required_ancestry(trajectory)
            if not required_nodes:
                print("No ancestry metadata found in step.extra.")
                continue
            print("--- required_ancestry ---")
            _print_tree(required_nodes)

        if args.validate:
            issues = _validate_trajectory_contract(trajectory)
            if issues:
                had_validation_errors = True
                print("--- validation ---")
                print(f"FAILED ({len(issues)} issues)")
                for issue in issues:
                    print(f"- {issue}")
            else:
                print("--- validation ---")
                print("PASSED")

    if args.validate and had_validation_errors:
        sys.exit(1)


if __name__ == "__main__":
    main()
