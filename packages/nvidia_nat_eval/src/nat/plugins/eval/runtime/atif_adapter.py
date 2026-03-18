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

from collections.abc import Mapping
from typing import Any

from nat.data_models.atif import ATIFTrajectory
from nat.data_models.evaluator import EvalInput
from nat.data_models.evaluator import EvalInputItem
from nat.plugins.eval.evaluator.atif_evaluator import AtifEvalSample
from nat.plugins.eval.evaluator.atif_evaluator import AtifEvalSampleList
from nat.utils.atif_converter import IntermediateStepToATIFConverter


class EvalAtifAdapter:
    """Build and cache ATIF trajectories for eval items."""

    def __init__(self, converter: IntermediateStepToATIFConverter | None = None) -> None:
        self._converter = converter or IntermediateStepToATIFConverter()
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
            samples.append(
                AtifEvalSample(
                    item_id=item.id,
                    trajectory=trajectory,
                    expected_output_obj=item.expected_output_obj,
                    output_obj=item.output_obj,
                    metadata={},
                ))
        return samples
