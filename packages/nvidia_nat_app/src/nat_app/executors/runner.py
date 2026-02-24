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
Speculative execution runner.

Provides ``run_speculation``, the core execution primitive for
the launch-await-decide-cancel-collect lifecycle shared by all framework
adapters that implement single-step speculation.

The runner dispatches resolution through ``plan.resolution.resolve()``
making it strategy-agnostic.

No framework imports -- uses only Python stdlib + nat_app execution primitives.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable
from collections.abc import Coroutine
from dataclasses import dataclass
from typing import Any

from nat_app.executors.execution_state import ExecutionState
from nat_app.speculation.plan import SpeculationPlan

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SpeculativeResult:
    """Outcome of a single speculative execution.

    Returned by ``run_speculation``.  Contains everything the caller
    needs for framework-specific post-processing (state merging,
    downstream cascading, etc.).
    """

    chosen_label: str
    """Decision label returned by ``get_decision``."""

    decision_result: Any
    """Raw result from executing the decision node."""

    chosen_results: dict[str, Any]
    """``{node_name: result}`` for speculative targets on the chosen path."""

    cancelled_nodes: frozenset[str]
    """Node names that were cancelled (unchosen paths)."""

    rerun_nodes: frozenset[str] = frozenset()
    """Nodes that need sequential re-execution (e.g. prediction misses)."""


async def run_speculation(
    plan: SpeculationPlan,
    execution_state: ExecutionState,
    *,
    run_node: Callable[[str], Coroutine[Any, Any, Any]],
    get_decision: Callable[[Any], str],
) -> SpeculativeResult:
    """Execute a decision node with speculative target launching.

    Handles the full launch-await-decide-cancel-collect lifecycle:

    1. Launch the decision node and all safe targets concurrently.
    2. Await the decision node result.
    3. Call *get_decision* to extract the decision label.
    4. Resolve via ``plan.resolution.resolve(label)`` to determine
       what to keep, cancel, and rerun.
    5. Cancel unchosen targets, collect chosen results.
    6. Update *execution_state* metrics throughout.

    Args:
        plan: Speculation plan produced by ``plan_speculation``.
        execution_state: Mutable execution state for metrics tracking.
        run_node: Framework-specific coroutine factory.  Called as
            ``await run_node(node_name)`` for the decision node and
            each target.
        get_decision: Extracts the decision label from the decision
            node result.  Called as ``get_decision(result) -> str``.

    Returns:
        A ``SpeculativeResult`` with the decision, all results,
        the set of cancelled nodes, and any nodes needing rerun.
    """
    decision_name = plan.decision_node

    # -- Launch decision node ----------------------------------------------
    decision_start = time.time()
    execution_state.tools_launched += 1
    execution_state.node_start_times[decision_name] = decision_start
    decision_task = asyncio.create_task(run_node(decision_name))

    # -- Launch speculative targets ----------------------------------------
    target_tasks: dict[str, asyncio.Task] = {}
    target_starts: dict[str, float] = {}

    logger.info(
        "Decision '%s': speculating %d targets: %s",
        decision_name,
        len(plan.targets_to_launch),
        sorted(plan.targets_to_launch),
    )

    for target_name in plan.targets_to_launch:
        start = time.time()
        target_starts[target_name] = start
        execution_state.tools_launched += 1
        execution_state.node_start_times[target_name] = start
        target_tasks[target_name] = asyncio.create_task(run_node(target_name))

    # -- Await decision node -----------------------------------------------
    try:
        decision_result = await decision_task
        decision_end = time.time()
        decision_duration = decision_end - decision_start

        execution_state.mark_node_completed(
            decision_name,
            decision_result if isinstance(decision_result, dict) else {},
        )
        execution_state.record_node_duration(decision_name, decision_duration)
        execution_state.record_timeline_event(decision_name, decision_start, decision_end)

        # -- Decide ------------------------------------------------------------
        chosen_label = get_decision(decision_result)
        execution_state.record_decision(decision_name, chosen_label, 1)
        logger.info("Decision '%s' chose: '%s'", decision_name, chosen_label)

        # -- Resolve via strategy policy ---------------------------------------
        resolution = plan.resolution.resolve(chosen_label)
    except Exception:
        for task in target_tasks.values():
            task.cancel()
        await asyncio.gather(*target_tasks.values(), return_exceptions=True)
        raise

    # -- Cancel unchosen ---------------------------------------------------
    actually_cancelled: set[str] = set()

    for name in resolution.cancel:
        task = target_tasks.get(name)
        if task is None:
            continue
        if not task.done():
            task.cancel()
        actually_cancelled.add(name)

    await asyncio.sleep(0)

    for name in actually_cancelled:
        task = target_tasks[name]
        if not task.done():
            try:
                await task
            except asyncio.CancelledError:
                pass
            except Exception:  # noqa: BLE001
                logger.exception("  '%s' raised while cancelling", name)
        elif not task.cancelled():
            try:
                task.result()
            except Exception:  # noqa: BLE001
                logger.exception("  '%s' raised after cancellation", name)
        cancel_time = time.time()
        execution_state.tools_cancelled += 1
        execution_state.record_timeline_event(
            name,
            target_starts.get(name, cancel_time),
            cancel_time,
            status="cancelled",
        )
        logger.info("  '%s' cancelled (unchosen)", name)

    # -- Collect chosen results --------------------------------------------
    chosen_results: dict[str, Any] = {}

    for name, task in target_tasks.items():
        if name in actually_cancelled:
            continue
        try:
            if not task.done():
                result = await task
            else:
                result = task.result()
        except Exception:  # noqa: BLE001
            logger.exception("  '%s' failed during speculation (will retry later)", name)
            end_time = time.time()
            start = target_starts[name]
            execution_state.mark_node_completed(name, {})
            execution_state.record_node_duration(name, end_time - start)
            execution_state.record_timeline_event(name, start, end_time, status="failed")
            continue

        end_time = time.time()
        start = target_starts[name]
        execution_state.mark_node_completed(
            name,
            result if isinstance(result, dict) else {},
        )
        execution_state.record_node_duration(name, end_time - start)
        execution_state.record_timeline_event(name, start, end_time)

        chosen_results[name] = result
        logger.info("  '%s' completed (chosen) in %.2fs", name, end_time - start)

    return SpeculativeResult(
        chosen_label=chosen_label,
        decision_result=decision_result,
        chosen_results=chosen_results,
        cancelled_nodes=frozenset(actually_cancelled),
        rerun_nodes=frozenset(resolution.rerun),
    )
