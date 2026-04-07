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
"""ATIF adapter utilities for eval runtime ingress.

This module provides a single-conversion adapter layer from ``EvalInputItem``
trajectory data to ``ATIFTrajectory`` objects. Runtime code uses this to avoid
per-evaluator conversion and to keep ATIF as the canonical internal trace shape.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

from nat.atif import ATIFTrajectory
from nat.data_models.evaluator import EvalInput
from nat.data_models.evaluator import EvalInputItem
from nat.plugins.eval.evaluator.atif_evaluator import AtifEvalSample
from nat.plugins.eval.evaluator.atif_evaluator import AtifEvalSampleList
from nat.utils.atif_converter import IntermediateStepToATIFConverter

logger = logging.getLogger(__name__)


class EvalAtifAdapter:
    """Build and cache ATIF trajectories for eval items."""

    def __init__(self,
                 converter: IntermediateStepToATIFConverter | None = None,
                 *,
                 allow_implicit_subagent_delegation: bool = False) -> None:
        self._converter = converter or IntermediateStepToATIFConverter(
            allow_implicit_subagent_delegation=allow_implicit_subagent_delegation)
        self._cache: dict[str, ATIFTrajectory] = {}

    @staticmethod
    def _cache_key(item_id: Any) -> str:
        item_type = type(item_id)
        return f"{item_type.__module__}.{item_type.__qualname__}:{item_id!r}"

    def _coerce_trajectory(self, value: Any) -> ATIFTrajectory:
        if isinstance(value, ATIFTrajectory):
            return value
        if isinstance(value, Mapping):
            return ATIFTrajectory.model_validate(value)
        raise TypeError(f"Unsupported ATIF trajectory payload type: {type(value)}")

    @staticmethod
    def _step_ancestry_ids(step) -> tuple[str | None, str | None]:
        """Return (function_id, parent_id) from a step's extra ancestry payload."""
        extra = step.extra or {}
        if not isinstance(extra, dict):
            return None, None
        ancestry = extra.get("ancestry")
        if not isinstance(ancestry, dict):
            return None, None
        function_id = ancestry.get("function_id")
        parent_id = ancestry.get("parent_id")
        if function_id is not None:
            function_id = str(function_id)
        if parent_id is not None:
            parent_id = str(parent_id)
        return function_id, parent_id

    @staticmethod
    def _collect_subtree_function_ids(trajectory: ATIFTrajectory, root_function_id: str) -> set[str]:
        """Collect function_ids in the ancestry subtree rooted at root_function_id."""
        parent_to_children: dict[str, set[str]] = {}
        for step in trajectory.steps:
            function_id, parent_id = EvalAtifAdapter._step_ancestry_ids(step)
            if not function_id or not parent_id:
                continue
            parent_to_children.setdefault(parent_id, set()).add(function_id)

        subtree_ids: set[str] = set()
        frontier = [root_function_id]
        while frontier:
            node = frontier.pop()
            if node in subtree_ids:
                continue
            subtree_ids.add(node)
            frontier.extend(parent_to_children.get(node, ()))
        return subtree_ids

    @staticmethod
    def _build_subtrajectory(trajectory: ATIFTrajectory, *, session_id: str, root_function_id: str) -> ATIFTrajectory | None:
        """Build a child trajectory view for a delegated function subtree."""
        subtree_ids = EvalAtifAdapter._collect_subtree_function_ids(trajectory, root_function_id)
        if not subtree_ids:
            return None

        selected_steps = []
        for step in trajectory.steps:
            function_id, _ = EvalAtifAdapter._step_ancestry_ids(step)
            if function_id in subtree_ids:
                selected_steps.append(step.model_copy(deep=True))
        if not selected_steps:
            return None

        reindexed_steps = [step.model_copy(update={"step_id": idx}) for idx, step in enumerate(selected_steps, start=1)]
        return ATIFTrajectory(
            schema_version=trajectory.schema_version,
            session_id=session_id,
            agent=trajectory.agent.model_copy(deep=True),
            steps=reindexed_steps,
            notes=trajectory.notes,
            extra={
                "parent_session_id": trajectory.session_id,
                "root_function_id": root_function_id,
            },
        )

    def _build_subagent_trajectory_map(self, trajectory: ATIFTrajectory, *, item_id: Any) -> dict[str, ATIFTrajectory]:
        """Build in-memory session_id -> trajectory map for subagent references."""
        by_session_id: dict[str, ATIFTrajectory] = {}
        unresolved: list[str] = []

        for step in trajectory.steps:
            if step.observation is None:
                continue
            for result in step.observation.results:
                refs = result.subagent_trajectory_ref or []
                for ref in refs:
                    session_id = ref.session_id
                    if not session_id:
                        continue
                    if session_id in by_session_id:
                        continue
                    root_function_id = None
                    if isinstance(ref.extra, dict):
                        child_function_id = ref.extra.get("child_function_id")
                        if child_function_id is not None:
                            root_function_id = str(child_function_id)
                    child_traj = (self._build_subtrajectory(
                        trajectory,
                        session_id=session_id,
                        root_function_id=root_function_id,
                    ) if root_function_id else None)
                    if child_traj is None:
                        unresolved.append(session_id)
                        # Degraded fallback keeps refs resolvable in-memory.
                        child_traj = trajectory.model_copy(
                            update={
                                "session_id": session_id,
                                "extra": {
                                    "parent_session_id": trajectory.session_id,
                                    "resolution_mode": "fallback_full_trajectory",
                                },
                            },
                            deep=True,
                        )
                    by_session_id[session_id] = child_traj

        if unresolved:
            logger.warning("ATIF subagent map fallback for item_id=%s; unresolved refs=%s", item_id, unresolved)
        return by_session_id

    def get_trajectory(self,
                       item: EvalInputItem,
                       prebuilt: ATIFTrajectory | Mapping[str, Any] | None = None) -> ATIFTrajectory:
        """Return cached ATIF trajectory for an eval item, converting at most once."""
        key = self._cache_key(item.id)
        if key in self._cache:
            return self._cache[key]

        if prebuilt is not None:
            trajectory = self._coerce_trajectory(prebuilt)
        else:
            trajectory = self._converter.convert(steps=item.trajectory, session_id=key)
        self._cache[key] = trajectory
        return trajectory

    def _ensure_cache(self,
                      eval_input: EvalInput,
                      prebuilt_trajectories: Mapping[str, ATIFTrajectory | Mapping[str, Any]] | None = None) -> None:
        """Populate cache for all eval items."""
        for item in eval_input.eval_input_items:
            prebuilt = None
            if prebuilt_trajectories is not None:
                # Prefer type-aware cache keys but allow legacy string keys.
                prebuilt = prebuilt_trajectories.get(self._cache_key(item.id))
                if prebuilt is None:
                    prebuilt = prebuilt_trajectories.get(str(item.id))
            self.get_trajectory(item=item, prebuilt=prebuilt)

    def build_samples(
            self,
            eval_input: EvalInput,
            prebuilt_trajectories: Mapping[str, ATIFTrajectory | Mapping[str, Any]] | None = None
    ) -> AtifEvalSampleList:
        """Build ATIF-native samples for all eval input items."""
        self._ensure_cache(eval_input=eval_input, prebuilt_trajectories=prebuilt_trajectories)
        samples: AtifEvalSampleList = []
        for item in eval_input.eval_input_items:
            trajectory = self._cache[self._cache_key(item.id)]
            subagent_map = self._build_subagent_trajectory_map(trajectory, item_id=item.id)
            samples.append(
                AtifEvalSample(
                    item_id=item.id,
                    trajectory=trajectory,
                    subagent_trajectories=subagent_map,
                    expected_output_obj=item.expected_output_obj,
                    output_obj=item.output_obj,
                    metadata={},
                ))
        return samples
