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
Simplified API for framework teams embedding nvidia-nat-app.

These functions accept standard Python data structures (dicts, lists, callables)
and return plain Python types (sets, lists, floats, strings).  No adapters,
compilers, or framework-specific classes are required.

Use these when integrating nvidia-nat-app within a framework's runtime:

    from nat_app.api import quick_optimize

    # Inside your framework's compile() or build() method:
    stages = quick_optimize(
        nodes={"a": fn_a, "b": fn_b, "c": fn_c},
        edges=[("a", "b"), ("a", "c")],
    )
    # stages = [{"a"}, {"b", "c"}]
    # Execute each stage's nodes in parallel, stages in sequence.

For the full compilation pipeline with custom stages and inter-stage
communication, use ``DefaultGraphCompiler`` instead.
"""

from __future__ import annotations

import asyncio
import copy
import statistics
import time
from collections.abc import Awaitable
from collections.abc import Callable
from typing import Any

from nat_app.compiler.default_graph_compiler import DefaultGraphCompiler
from nat_app.graph.factory import build_graph_and_adapter
from nat_app.graph.scheduling import compute_branch_info
from nat_app.graph.static_analysis import analyze_function_ast
from nat_app.graph.topology import analyze_graph_topology
from nat_app.speculation.plan import SpeculationPlan  # noqa: F401  # pylint: disable=unused-import
from nat_app.speculation.plan import partition_targets  # noqa: F401  # pylint: disable=unused-import
from nat_app.speculation.plan import plan_speculation  # noqa: F401  # pylint: disable=unused-import


def quick_optimize(
    nodes: dict[str, Callable | None],
    edges: list[tuple[str, str]],
    entry: str | None = None,
    conditional_edges: dict[str, dict[str, str | list[str]]] | None = None,
    self_state_attrs: dict[str, str] | None = None,
) -> list[set[str]]:
    """Compute parallel execution stages from raw graph data.

    This is the primary entry point for framework teams.  It takes your
    graph as plain Python data, runs the full optimization pipeline
    (AST analysis, edge classification, scheduling), and returns a list
    of parallel stages.

    Args:
        nodes: Mapping of node name to callable function (or None if the
            function is unavailable for AST analysis).
        edges: List of ``(source, target)`` dependency edges.
        entry: Entry point node name.  Defaults to the first key in ``nodes``.
        conditional_edges: Optional router/conditional edges.  Maps a router
            node name to ``{target_name: target_name}`` for each branch.
        self_state_attrs: For class methods that access state through
            ``self.X``, maps the attribute name to an object namespace.
            For example, ``{"state": "state"}`` tells the AST analyzer that
            ``self.state["key"]`` is a state read/write.

    Returns:
        A list of sets where each set contains node names that can execute
        in parallel.  Execute stages in order, all nodes within a stage
        concurrently.

    Example:

        stages = quick_optimize(
            nodes={"parse": parse_fn, "research_a": fn_a, "research_b": fn_b, "synthesize": fn_c},
            edges=[("parse", "research_a"), ("parse", "research_b"),
                   ("research_a", "synthesize"), ("research_b", "synthesize")],
        )
        # Returns: [{"parse"}, {"research_a", "research_b"}, {"synthesize"}]
    """
    graph, adapter = build_graph_and_adapter(nodes, edges, entry, conditional_edges, self_state_attrs)

    compiler = DefaultGraphCompiler(adapter)
    context = compiler.compile(graph)
    return context.optimized_order or []


def analyze_function(
    func: Callable,
    self_state_attrs: dict[str, str] | None = None,
    max_recursion_depth: int = 5,
) -> dict[str, Any]:
    """Analyze a function's state reads and writes via AST.

    Returns plain Python types so framework teams can use the data
    directly without importing nvidia-nat-app internal types.

    Args:
        func: The function to analyze.
        self_state_attrs: For class methods, maps ``self.X`` attribute names
            to object namespaces (e.g. ``{"state": "state"}``).
        max_recursion_depth: Max call depth when following callees. Default 5.

    Returns:
        A dict with keys:

        - ``reads``: ``set[str]`` of state keys the function reads
        - ``writes``: ``set[str]`` of state keys the function writes/mutates
        - ``confidence``: ``"full"`` | ``"partial"`` | ``"opaque"`` indicating
          analysis reliability
        - ``warnings``: ``list[str]`` of any issues encountered during analysis
        - ``source_available``: ``bool`` whether source code was found

    Example:

        info = analyze_function(my_node_fn)
        if info["confidence"] == "full":
            print(f"Reads: {info['reads']}, Writes: {info['writes']}")
    """
    result = analyze_function_ast(
        func,
        self_state_attrs=self_state_attrs,
        max_recursion_depth=max_recursion_depth,
    )

    reads = result.reads.all_fields_flat if result.source_available else set()
    writes = result.all_writes.all_fields_flat if result.source_available else set()

    if not result.source_available:
        confidence = "opaque"
    else:
        uncertainty_flags = (result.has_dynamic_keys or result.has_unresolved_calls or result.recursion_depth_hit
                             or result.has_dynamic_exec or result.has_closure_write or result.has_global_write
                             or result.has_unknown_attr_access or result.has_return_lambda_mutates_state
                             or result.has_dynamic_attr)
        warnings_without_writes = not writes and result.warnings
        if uncertainty_flags or warnings_without_writes:
            confidence = "partial"
        else:
            confidence = "full"

    return {
        "reads": reads,
        "writes": writes,
        "confidence": confidence,
        "warnings": list(result.warnings),
        "source_available": result.source_available,
    }


def classify_edge(
    source_func: Callable,
    target_func: Callable,
    self_state_attrs: dict[str, str] | None = None,
) -> str:
    """Check if a dependency edge between two functions is necessary.

    Analyzes both functions via AST.  An edge is "necessary" if the source
    writes state keys that the target reads.  An edge is "unnecessary" if
    there is no read/write overlap.  Returns "unknown" if either function
    cannot be fully analyzed (i.e. confidence is not "full").

    Args:
        source_func: The upstream function.
        target_func: The downstream function.
        self_state_attrs: For class methods, maps ``self.X`` -> object namespace.

    Returns:
        One of: ``"necessary"``, ``"unnecessary"``, or ``"unknown"``.

    Example:

        result = classify_edge(step_a_fn, step_b_fn)
        if result == "unnecessary":
            # step_b doesn't read step_a's outputs -- they can run in parallel
            ...
    """
    src = analyze_function(source_func, self_state_attrs=self_state_attrs)
    tgt = analyze_function(target_func, self_state_attrs=self_state_attrs)

    if src["confidence"] != "full" or tgt["confidence"] != "full":
        return "unknown"

    overlap = src["writes"] & tgt["reads"]
    return "necessary" if overlap else "unnecessary"


def find_parallel_stages(
    nodes: dict[str, Callable | None],
    edges: list[tuple[str, str]],
    self_state_attrs: dict[str, str] | None = None,
) -> tuple[list[set[str]], dict[str, dict[str, Any]]]:
    """Compute parallel stages and per-node analysis details.

    Like ``quick_optimize`` but also returns the per-node analysis
    data so framework teams can inspect what the optimizer discovered.

    Args:
        nodes: Mapping of node name to callable (or None).
        edges: List of ``(source, target)`` edges.
        self_state_attrs: For class methods, maps ``self.X`` -> object namespace.

    Returns:
        A tuple of ``(stages, node_info)`` where:

        - ``stages``: ``list[set[str]]`` -- parallel stage groupings
        - ``node_info``: ``dict[str, dict]`` -- per-node analysis with keys
          ``reads``, ``writes``, ``confidence``, ``warnings``

    Example:

        stages, info = find_parallel_stages(
            nodes={"a": fn_a, "b": fn_b},
            edges=[("a", "b")],
        )
        print(f"Stage plan: {stages}")
        for name, analysis in info.items():
            print(f"  {name}: reads={analysis['reads']}, writes={analysis['writes']}")
    """
    graph, adapter = build_graph_and_adapter(nodes, edges, self_state_attrs=self_state_attrs)

    compiler = DefaultGraphCompiler(adapter)
    context = compiler.compile(graph)

    stages = context.optimized_order or []

    node_info: dict[str, dict[str, Any]] = {}
    node_analyses = context.node_analyses or {}
    for name, analysis in node_analyses.items():
        node_info[name] = {
            "reads": analysis.reads.all_fields_flat,
            "writes": analysis.mutations.all_fields_flat,
            "confidence": analysis.confidence,
            "warnings": list(analysis.warnings),
        }

    return stages, node_info


# ---------------------------------------------------------------------------
# Benchmarking
# ---------------------------------------------------------------------------


async def benchmark(
    nodes: dict[str, Callable | None],
    edges: list[tuple[str, str]],
    execute_node: Callable[[str, dict[str, Any]], Awaitable[dict[str, Any]]],
    strategies: dict[str, Callable[[dict[str, Any]], Awaitable[dict[str, Any]]]] | None = None,
    initial_state: dict[str, Any] | None = None,
    n_runs: int = 3,
    **optimize_kwargs: Any,
) -> dict[str, Any]:
    """Benchmark sequential vs. optimized execution strategies.

    Runs a sequential baseline (nodes one-at-a-time), a parallel-stages
    baseline (using ``quick_optimize`` output), and any number of
    custom strategy executors.  Each variant runs *n_runs* times and the
    median wall-clock time is reported.

    Args:
        nodes: Mapping of node name to callable (or None).
        edges: List of ``(source, target)`` dependency edges.
        execute_node: Async callable ``(node_name, state) -> dict``. Must return
            a dict (updated state); non-dict returns raise TypeError. Used for
            the built-in sequential and parallel baselines.
        strategies: Optional dict of ``{name: async_executor}``.  Each
            executor is called as ``await executor(copy.deepcopy(state))``
            and is responsible for its own execution logic.
        initial_state: Starting state dict (deep-copied before each run).
        n_runs: Number of repetitions; the median is reported.
        **optimize_kwargs: Forwarded to ``quick_optimize``.

    Returns:
        A dict with keys:

        - ``sequential_ms``, ``parallel_ms``, ``parallel_speedup``
        - ``strategies`` -- per-strategy timing results
        - ``stages`` -- the parallel stage plan
        - ``static_estimate``, ``n_runs``
        - ``outputs`` -- last-run output for each baseline and strategy

    Notes:
        The parallel baseline merges stage results with ``dict.update`` (last-write-wins).
        The scheduler guarantees that no two nodes in the same parallel stage write
        overlapping non-reducer state keys, so the merge is correct for those keys.
        For reducer fields (e.g. LangGraph ``messages``), multiple nodes may write
        in parallel; framework-specific merge semantics may differ from ``update``.

    Example:

        results = await benchmark(
            nodes={"a": fn_a, "b": fn_b, "c": fn_c},
            edges=[("a", "b"), ("a", "c")],
            execute_node=run_node,
            strategies={"speculative": my_spec_runner},
        )
        print(f"Parallel speedup: {results['parallel_speedup']:.2f}x")
        seq_output = results["outputs"]["sequential"]
    """
    state = initial_state or {}
    stages = quick_optimize(nodes=nodes, edges=edges, **optimize_kwargs)

    topo_order: list[str] = []
    for stage in stages:
        topo_order.extend(sorted(stage))

    static_estimate = len(topo_order) / len(stages) if stages else 1.0

    async def _run_sequential() -> tuple[float, dict]:
        s = copy.deepcopy(state)
        t0 = time.perf_counter()
        for name in topo_order:
            result = await execute_node(name, s)
            if not isinstance(result, dict):
                raise TypeError(f"execute_node must return a dict, got {type(result).__name__} from node {name!r}")
            s = result
        return (time.perf_counter() - t0) * 1000, s

    async def _run_parallel() -> tuple[float, dict]:
        s = copy.deepcopy(state)
        t0 = time.perf_counter()
        for stage in stages:
            stage_results = await asyncio.gather(*(execute_node(name, copy.deepcopy(s)) for name in sorted(stage)))
            for name, r in zip(sorted(stage), stage_results, strict=True):
                if not isinstance(r, dict):
                    raise TypeError(f"execute_node must return a dict, got {type(r).__name__} from node {name!r}")
                # Scheduler ensures no write-write conflicts in parallel stages.
                s.update(r)
        return (time.perf_counter() - t0) * 1000, s

    seq_output: dict = {}
    par_output: dict = {}
    seq_times: list[float] = []
    par_times: list[float] = []
    for _ in range(n_runs):
        elapsed, seq_output = await _run_sequential()
        seq_times.append(elapsed)
    for _ in range(n_runs):
        elapsed, par_output = await _run_parallel()
        par_times.append(elapsed)

    seq_median = statistics.median(seq_times)
    par_median = statistics.median(par_times)

    outputs: dict[str, Any] = {
        "sequential": seq_output,
        "parallel": par_output,
    }

    result: dict[str, Any] = {
        "sequential_ms": round(seq_median, 2),
        "parallel_ms": round(par_median, 2),
        "parallel_speedup": round(seq_median / par_median, 2) if par_median > 0 else float("inf"),
        "strategies": {},
        "stages": stages,
        "static_estimate": round(static_estimate, 2),
        "n_runs": n_runs,
        "outputs": outputs,
    }

    if strategies:
        for name, executor in strategies.items():
            strat_times: list[float] = []
            strat_output: Any = None
            for _ in range(n_runs):
                s = copy.deepcopy(state)
                t0 = time.perf_counter()
                ret = await executor(s)
                elapsed = (time.perf_counter() - t0) * 1000
                strat_times.append(elapsed)
                strat_output = ret if ret is not None else s
            strat_median = statistics.median(strat_times)
            result["strategies"][name] = {
                "median_ms": round(strat_median, 2),
                "speedup_vs_sequential": round(seq_median / strat_median, 2) if strat_median > 0 else float("inf"),
                "speedup_vs_parallel": round(par_median / strat_median, 2) if strat_median > 0 else float("inf"),
            }
            outputs[name] = strat_output

    return result


# ---------------------------------------------------------------------------
# Speculative opportunity analysis
# ---------------------------------------------------------------------------


def speculative_opportunities(
    nodes: dict[str, Callable | None],
    edges: list[tuple[str, str]],
    conditional_edges: dict[str, dict[str, str | list[str]]] | None = None,
    self_state_attrs: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    """Identify speculative execution opportunities in a graph.

    Analyzes routers and their branch structures to find places where
    speculative execution could save time by launching branch targets
    before the router decides.

    Args:
        nodes: Mapping of node name to callable (or None).
        edges: List of ``(source, target)`` dependency edges.
        conditional_edges: Router/conditional edges.  Maps a router node
            to ``{branch_name: target_node}`` for each branch.
        self_state_attrs: For class methods, maps ``self.X`` -> namespace.

    Returns:
        A list of opportunity dicts, one per decision node, each containing:

        - ``decision_node``: name of the decision node
        - ``branches``: ``{target: [exclusive_nodes]}``
        - ``merge_nodes``: nodes shared across branches
        - ``speculatable_nodes``: count of nodes that could run speculatively
        - ``max_branch_depth``: longest exclusive-branch path
        - ``is_cycle_exit``: whether the decision node also controls a loop

    Example:

        opps = speculative_opportunities(
            nodes={"router": route_fn, "a": fn_a, "b": fn_b, "merge": fn_m},
            edges=[("router", "a"), ("router", "b"), ("a", "merge"), ("b", "merge")],
            conditional_edges={"router": {"left": "a", "right": "b"}},
        )
        for opp in opps:
            print(f"Decision node {opp['decision_node']}: {opp['speculatable_nodes']} speculatable nodes")
    """
    graph, adapter = build_graph_and_adapter(
        nodes, edges, conditional_edges=conditional_edges,
        self_state_attrs=self_state_attrs,
    )

    topology = analyze_graph_topology(graph)
    if not topology.routers:
        return []

    branch_info = compute_branch_info(graph, topology)

    results: list[dict[str, Any]] = []
    router_lookup = {r.node: r for r in topology.routers}

    for rnode, binfo in branch_info.items():
        router = router_lookup.get(rnode)
        branches_plain: dict[str, list[str]] = {label: sorted(exclusive) for label, exclusive in binfo.branches.items()}
        max_depth = max((len(v) for v in branches_plain.values()), default=0)
        speculatable = sum(len(v) for v in branches_plain.values())

        results.append({
            "decision_node": rnode,
            "branches": branches_plain,
            "merge_nodes": sorted(binfo.merge_nodes),
            "speculatable_nodes": speculatable,
            "max_branch_depth": max_depth,
            "is_cycle_exit": router.is_cycle_exit if router else False,
        })

    return results
