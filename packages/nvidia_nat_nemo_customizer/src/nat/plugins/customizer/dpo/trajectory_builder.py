# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
DPO (Direct Preference Optimization) Trajectory Builder.

This module provides a trajectory builder that collects preference data from
workflows that produce TTC_END intermediate steps with TTCEventData.

The builder:
1. Runs evaluation to collect intermediate steps
2. Filters for TTC_END steps with the configured name
3. Extracts data from TTCEventData (turn_id, candidate_index, score, input, output)
4. Groups candidates by turn_id
5. Generates preference pairs based on score differences
6. Builds trajectories with DPOItem episodes for DPO training
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from dataclasses import field
from datetime import datetime
from pathlib import Path
from typing import Any

from nat.data_models.finetuning import DPOItem
from nat.data_models.finetuning import OpenAIMessage
from nat.data_models.finetuning import Trajectory
from nat.data_models.finetuning import TrajectoryCollection
from nat.data_models.intermediate_step import IntermediateStep
from nat.data_models.intermediate_step import IntermediateStepCategory
from nat.data_models.intermediate_step import IntermediateStepType
from nat.data_models.intermediate_step import StreamEventData
from nat.data_models.intermediate_step import TTCEventData
from nat.eval.config import EvaluationRunOutput
from nat.eval.evaluator.evaluator_model import EvalInputItem
from nat.finetuning.interfaces.trajectory_builder import TrajectoryBuilder

from .config import DPOTrajectoryBuilderConfig

logger = logging.getLogger(__name__)

# Type alias for prompt which can be string or list of OpenAI messages
PromptType = list[OpenAIMessage] | str

# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class CandidateStep:
    """
    Parsed candidate from a TTC intermediate step.

    Represents a single candidate response that was generated and scored
    for a particular turn in the workflow.
    """

    example_id: str
    """Unique identifier for the dataset example."""

    turn_id: str
    """Identifier for the turn (groups candidates competing for the same prompt)."""

    candidate_index: int
    """Index of this candidate within the turn."""

    prompt: PromptType
    """Input prompt that produced this response (string or list of OpenAIMessage)."""

    response: str
    """Model's response/completion."""

    score: float
    """Score assigned to this candidate (higher is better)."""

    raw_metadata: dict[str, Any] = field(default_factory=dict)
    """Original metadata from the intermediate step."""


@dataclass
class PreferencePair:
    """
    A preference pair for DPO training.

    Represents a single (prompt, chosen, rejected) triple where the chosen
    response has a higher score than the rejected response.
    """

    example_id: str
    """Unique identifier for the dataset example."""

    turn_id: str
    """Identifier for the turn."""

    prompt: PromptType
    """Input prompt (same for both responses)."""

    chosen_response: str
    """Response that was preferred (higher score)."""

    rejected_response: str
    """Response that was not preferred (lower score)."""

    chosen_score: float
    """Score of the chosen response."""

    rejected_score: float
    """Score of the rejected response."""

    score_diff: float
    """Difference between chosen and rejected scores."""

    chosen_index: int
    """Candidate index of the chosen response."""

    rejected_index: int
    """Candidate index of the rejected response."""

    metadata: dict[str, Any] = field(default_factory=dict)
    """Additional metadata for the pair."""


# =============================================================================
# DPO Trajectory Builder
# =============================================================================


class DPOTrajectoryBuilder(TrajectoryBuilder):
    """
    Trajectory builder for DPO (Direct Preference Optimization) training.

    This builder collects preference pairs from workflows that produce TTC_END
    intermediate steps with TTCEventData. It uses the structured data model
    to extract turn_id, candidate_index, score, input (prompt), and output.

    Key features:
    - Uses TTCEventData model directly (no brittle dictionary key configuration)
    - Supports prompts as strings or list of OpenAIMessage
    - Exhaustive or best-vs-worst pair generation modes
    - Configurable score difference filtering
    - Grouping by example for curriculum learning
    - Builds trajectories with DPOItem episodes

    Example workflow integration::

        trajectory_builders:
          dpo_builder:
            _type: dpo_traj_builder
            ttc_step_name: dpo_candidate_move
            exhaustive_pairs: true
            min_score_diff: 0.05
    """

    def __init__(self, trajectory_builder_config: DPOTrajectoryBuilderConfig):
        """
        Initialize the DPO Trajectory Builder.

        Args:
            trajectory_builder_config: Configuration for the builder.
        """
        super().__init__(trajectory_builder_config=trajectory_builder_config)
        self.config: DPOTrajectoryBuilderConfig = trajectory_builder_config
        self.evaluation_runs: dict[str, asyncio.Task[EvaluationRunOutput]] = {}

        # Metrics tracking
        self._metrics: dict[str, Any] = {}

    # =========================================================================
    # TrajectoryBuilder Interface Implementation
    # =========================================================================

    async def start_run(self, run_id: str, meta: dict | None = None) -> None:
        """
        Start a single evaluation run to collect intermediate steps.

        Args:
            run_id: Unique identifier for this run.
            meta: Optional metadata for the run.

        Raises:
            ValueError: If a run with this ID is already in progress.
        """
        if run_id in self.evaluation_runs:
            raise ValueError(f"Run {run_id} is already in progress.")

        logger.info("Starting DPO evaluation run: %s", run_id)
        logger.info(
            "Configuration: step_name=%s, exhaustive=%s, min_diff=%.3f",
            self.config.ttc_step_name,
            self.config.exhaustive_pairs,
            self.config.min_score_diff,
        )

        # Create evaluation task
        task = asyncio.create_task(self.run_eval(), name=f"dpo-eval-{run_id}")

        def _on_done(t: asyncio.Task[EvaluationRunOutput]) -> None:
            if t.cancelled():
                logger.info("DPO evaluation run %s was cancelled.", run_id)
            elif exc := t.exception():
                logger.error("DPO evaluation run %s failed: %s", run_id, exc)
            else:
                logger.info("DPO evaluation run %s completed successfully.", run_id)

        task.add_done_callback(_on_done)
        self.evaluation_runs[run_id] = task

    async def finalize(self, run_id: str, meta: dict | None = None) -> TrajectoryCollection:
        """
        Wait for evaluation, collect TTC steps, and build DPO trajectories.

        This method:
        1. Waits for the evaluation run to complete
        2. Collects and groups candidates by turn_id using TTCEventData
        3. Generates preference pairs
        4. Builds trajectories with DPOItem episodes
        5. Groups trajectories by example for curriculum learning

        Args:
            run_id: Unique identifier for the run.
            meta: Optional metadata for the run.

        Returns:
            TrajectoryCollection with DPO preference trajectories.

        Raises:
            ValueError: If no run with this ID exists.
        """
        if run_id not in self.evaluation_runs:
            raise ValueError(f"No evaluation run found for run_id: {run_id}")

        # Wait for evaluation to complete
        logger.info("Waiting for DPO evaluation run %s to complete...", run_id)
        eval_result = await self.evaluation_runs[run_id]

        # Initialize metrics
        self._metrics = {
            "run_id": run_id,
            "total_examples": 0,
            "total_turns": 0,
            "total_candidates": 0,
            "total_pairs": 0,
            "total_trajectories": 0,
            "skipped_single_candidate": 0,
            "skipped_score_diff": 0,
        }

        # Step 1: Collect and group candidates
        candidates_by_turn = self._collect_candidates(eval_result)
        self._metrics["total_turns"] = len(candidates_by_turn)

        if not candidates_by_turn:
            logger.warning("No candidate steps found for run_id: %s", run_id)
            del self.evaluation_runs[run_id]
            return TrajectoryCollection(trajectories=[], run_id=run_id)

        # Step 2: Generate preference pairs
        pairs = self._generate_preference_pairs(candidates_by_turn)
        self._metrics["total_pairs"] = len(pairs)

        if not pairs:
            logger.warning("No preference pairs generated for run_id: %s", run_id)
            del self.evaluation_runs[run_id]
            return TrajectoryCollection(trajectories=[], run_id=run_id)

        # Step 3: Build trajectories with DPOItem episodes
        trajectories = self._build_trajectories(pairs)
        self._metrics["total_trajectories"] = len(trajectories)

        # Step 4: Group by example for curriculum learning
        grouped = self._group_by_example(trajectories)
        self._metrics["total_examples"] = len(grouped)

        # Log summary
        logger.info(
            "DPO trajectory building complete for run %s: "
            "%d examples, %d turns, %d candidates, %d pairs, %d trajectories",
            run_id,
            self._metrics["total_examples"],
            self._metrics["total_turns"],
            self._metrics["total_candidates"],
            self._metrics["total_pairs"],
            self._metrics["total_trajectories"],
        )

        if self._metrics["skipped_single_candidate"] > 0:
            logger.info(
                "Skipped %d turns with single candidate (no preference signal)",
                self._metrics["skipped_single_candidate"],
            )

        if self._metrics["skipped_score_diff"] > 0:
            logger.info(
                "Skipped %d pairs with score diff < %.3f",
                self._metrics["skipped_score_diff"],
                self.config.min_score_diff,
            )

        # Cleanup
        del self.evaluation_runs[run_id]

        return TrajectoryCollection(trajectories=grouped, run_id=run_id)

    def log_progress(self, run_id: str, metrics: dict[str, Any], output_dir: str | None = None) -> None:
        """
        Log trajectory building progress.

        Args:
            run_id: The training run ID.
            metrics: Dictionary of metrics to log.
            output_dir: Optional output directory override.
        """
        # Use default output directory if not provided
        out_dir = (Path(output_dir) if output_dir else Path("./.tmp/nat/finetuning/dpo_trajectory_builder"))
        out_dir.mkdir(parents=True, exist_ok=True)

        # Create log file
        log_file = out_dir / f"dpo_trajectory_builder_{run_id}.jsonl"

        # Prepare log entry
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "run_id": run_id,
            "config": {
                "ttc_step_name": self.config.ttc_step_name,
                "exhaustive_pairs": self.config.exhaustive_pairs,
                "min_score_diff": self.config.min_score_diff,
                "max_pairs_per_turn": self.config.max_pairs_per_turn,
            },
            **self._metrics,
            **metrics,
        }

        # Append to log file
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry) + "\n")

        logger.debug(
            "DPO trajectory builder progress logged for run %s: %d pairs",
            run_id,
            self._metrics.get("total_pairs", 0),
        )

    # =========================================================================
    # Internal Methods
    # =========================================================================

    def _collect_candidates(self, eval_result: EvaluationRunOutput) -> dict[str, list[CandidateStep]]:
        """
        Extract TTC_END intermediate steps and group by turn_id.

        This method:
        1. Iterates through all evaluation input items
        2. Filters for TTC_END steps with the configured name
        3. Extracts data from TTCEventData model directly
        4. Groups candidates by (example_id, turn_id)

        Args:
            eval_result: The evaluation run output.

        Returns:
            Dictionary mapping turn keys to lists of candidates.
        """
        candidates_by_turn: dict[str, list[CandidateStep]] = {}

        # Create mapping of example ID to input item
        input_items_map: dict[str, EvalInputItem] = {item.id: item for item in eval_result.eval_input.eval_input_items}

        for example_id, input_item in input_items_map.items():
            # Filter for TTC_END steps with matching name
            for step in input_item.trajectory:
                if not self._is_target_step(step):
                    continue

                # Parse candidate from TTCEventData
                candidate = self._parse_candidate(example_id, step)
                if candidate is None:
                    continue

                self._metrics["total_candidates"] = (self._metrics.get("total_candidates", 0) + 1)

                # Group by (example_id, turn_id)
                turn_key = f"{example_id}::{candidate.turn_id}"
                if turn_key not in candidates_by_turn:
                    candidates_by_turn[turn_key] = []
                candidates_by_turn[turn_key].append(candidate)

        logger.debug(
            "Collected %d candidates across %d turns",
            self._metrics.get("total_candidates", 0),
            len(candidates_by_turn),
        )

        return candidates_by_turn

    def _is_target_step(self, step: IntermediateStep) -> bool:
        """
        Check if an intermediate step is a target TTC step.

        Args:
            step: The intermediate step to check.

        Returns:
            True if this is a TTC_END step with the configured name.
        """
        return (step.event_category == IntermediateStepCategory.TTC and step.event_type == IntermediateStepType.TTC_END
                and step.payload.name == self.config.ttc_step_name)

    def _parse_candidate(self, example_id: str, step: IntermediateStep) -> CandidateStep | None:
        """
        Parse a CandidateStep from a TTC intermediate step using TTCEventData.

        Args:
            example_id: The example ID this step belongs to.
            step: The intermediate step to parse.

        Returns:
            CandidateStep if parsing succeeds, None otherwise.
        """
        # Get TTCEventData from step.payload.data
        data = step.payload.data
        if data is None:
            logger.warning("Step has no data field, skipping: %s", step.payload.UUID)
            return None

        # Validate that we have TTCEventData (or compatible dict/StreamEventData)
        # NOTE: When IntermediateStepPayload is serialized/deserialized, TTCEventData
        # becomes StreamEventData because the data field is typed as StreamEventData.
        # The TTC fields are preserved as extra fields due to extra="allow".
        if isinstance(data, TTCEventData):
            ttc_data = data
        elif isinstance(data, StreamEventData):
            # TTCEventData may have been deserialized as StreamEventData
            # Try to construct TTCEventData from the model dump
            try:
                data_dict = data.model_dump()
                ttc_data = TTCEventData(**data_dict)
            except Exception as e:
                logger.warning("Failed to parse TTCEventData from StreamEventData: %s", e)
                return None
        elif isinstance(data, dict):
            # Try to parse as TTCEventData
            try:
                ttc_data = TTCEventData(**data)
            except Exception as e:
                logger.warning("Failed to parse TTCEventData from dict: %s", e)
                return None
        else:
            logger.warning("Unexpected data type %s, expected TTCEventData", type(data))
            return None

        # Extract required fields from TTCEventData
        try:
            turn_id = ttc_data.turn_id
            if turn_id is None:
                logger.warning(
                    "TTCEventData missing turn_id, skipping: %s",
                    step.payload.UUID,
                )
                return None

            score = ttc_data.score
            if score is None:
                logger.warning(
                    "TTCEventData missing score, skipping: %s",
                    step.payload.UUID,
                )
                return None

            candidate_index = ttc_data.candidate_index or 0

            # Get prompt from TTCEventData.input
            # This can be a string or list of OpenAIMessage
            prompt = self._extract_prompt(ttc_data.input)

            # Get response from TTCEventData.output
            response = str(ttc_data.output) if ttc_data.output else ""

            # Get raw metadata for additional context
            raw_metadata = {}
            if step.payload.metadata:
                if hasattr(step.payload.metadata, "model_dump"):
                    raw_metadata = step.payload.metadata.model_dump()
                elif isinstance(step.payload.metadata, dict):
                    raw_metadata = step.payload.metadata

            return CandidateStep(
                example_id=str(example_id),
                turn_id=str(turn_id),
                candidate_index=int(candidate_index),
                prompt=prompt,
                response=response,
                score=float(score),
                raw_metadata=raw_metadata,
            )

        except (TypeError, ValueError) as e:
            logger.warning(
                "Failed to parse candidate from step %s: %s",
                step.payload.UUID,
                e,
            )
            return None

    def _extract_prompt(self, input_data: Any) -> PromptType:
        """
        Extract prompt from TTCEventData.input.

        Handles both string prompts and list of OpenAIMessage.

        Args:
            input_data: The input field from TTCEventData.

        Returns:
            String prompt or list of OpenAIMessage.
        """
        if input_data is None:
            return ""

        if isinstance(input_data, str):
            return input_data

        if isinstance(input_data, list):
            # Try to convert to list of OpenAIMessage
            messages: list[OpenAIMessage] = []
            for item in input_data:
                if isinstance(item, OpenAIMessage):
                    messages.append(item)
                elif isinstance(item, dict):
                    # Try to parse as OpenAIMessage
                    try:
                        messages.append(OpenAIMessage(**item))
                    except Exception:
                        # If parsing fails, convert entire input to string
                        return str(input_data)
                else:
                    # Unknown type, convert to string
                    return str(input_data)
            return messages

        # Fallback: convert to string
        return str(input_data)

    def _generate_preference_pairs(self, candidates_by_turn: dict[str, list[CandidateStep]]) -> list[PreferencePair]:
        """
        Generate preference pairs from grouped candidates.

        If exhaustive_pairs=True:
            For candidates [A, B, C] with scores [0.9, 0.7, 0.5]:
            Pairs: (A>B), (A>C), (B>C) - all pairwise comparisons

        If exhaustive_pairs=False:
            For candidates [A, B, C] with scores [0.9, 0.7, 0.5]:
            Pairs: (A>C) only - best vs worst

        Args:
            candidates_by_turn: Dictionary mapping turn keys to candidate lists.

        Returns:
            List of preference pairs.
        """
        all_pairs: list[PreferencePair] = []

        for turn_key, candidates in candidates_by_turn.items():
            # Check if we have enough candidates
            if len(candidates) < 2:
                if self.config.require_multiple_candidates:
                    self._metrics["skipped_single_candidate"] = (self._metrics.get("skipped_single_candidate", 0) + 1)
                    logger.debug("Skipping turn %s with single candidate", turn_key)
                    continue

            # Sort candidates by score (descending)
            sorted_candidates = sorted(candidates, key=lambda c: c.score, reverse=True)

            if self.config.exhaustive_pairs:
                pairs = self._generate_exhaustive_pairs(sorted_candidates)
            else:
                pairs = self._generate_best_vs_worst_pair(sorted_candidates)

            all_pairs.extend(pairs)

        logger.debug("Generated %d preference pairs", len(all_pairs))
        return all_pairs

    def _generate_exhaustive_pairs(self, sorted_candidates: list[CandidateStep]) -> list[PreferencePair]:
        """
        Generate all pairwise comparisons where score(chosen) > score(rejected).

        Args:
            sorted_candidates: Candidates sorted by score (descending).

        Returns:
            List of preference pairs, sorted by score difference (descending).
        """
        pairs: list[PreferencePair] = []

        for i, chosen in enumerate(sorted_candidates):
            for rejected in sorted_candidates[i + 1:]:
                score_diff = chosen.score - rejected.score

                # Apply minimum score difference filter
                if score_diff < self.config.min_score_diff:
                    self._metrics["skipped_score_diff"] = (self._metrics.get("skipped_score_diff", 0) + 1)
                    continue

                pairs.append(
                    PreferencePair(
                        example_id=chosen.example_id,
                        turn_id=chosen.turn_id,
                        prompt=chosen.prompt,
                        chosen_response=chosen.response,
                        rejected_response=rejected.response,
                        chosen_score=chosen.score,
                        rejected_score=rejected.score,
                        score_diff=score_diff,
                        chosen_index=chosen.candidate_index,
                        rejected_index=rejected.candidate_index,
                        metadata={
                            "chosen_raw_metadata": chosen.raw_metadata,
                            "rejected_raw_metadata": rejected.raw_metadata,
                        },
                    ))

        # Sort by score difference (highest first) and apply limit
        pairs.sort(key=lambda p: p.score_diff, reverse=True)

        if self.config.max_pairs_per_turn is not None:
            pairs = pairs[:self.config.max_pairs_per_turn]

        return pairs

    def _generate_best_vs_worst_pair(self, sorted_candidates: list[CandidateStep]) -> list[PreferencePair]:
        """
        Generate a single pair: best candidate vs worst candidate.

        Args:
            sorted_candidates: Candidates sorted by score (descending).

        Returns:
            List with at most one preference pair.
        """
        if len(sorted_candidates) < 2:
            return []

        chosen = sorted_candidates[0]  # Best
        rejected = sorted_candidates[-1]  # Worst

        score_diff = chosen.score - rejected.score

        # Apply minimum score difference filter
        if score_diff < self.config.min_score_diff:
            self._metrics["skipped_score_diff"] = (self._metrics.get("skipped_score_diff", 0) + 1)
            return []

        return [
            PreferencePair(
                example_id=chosen.example_id,
                turn_id=chosen.turn_id,
                prompt=chosen.prompt,
                chosen_response=chosen.response,
                rejected_response=rejected.response,
                chosen_score=chosen.score,
                rejected_score=rejected.score,
                score_diff=score_diff,
                chosen_index=chosen.candidate_index,
                rejected_index=rejected.candidate_index,
                metadata={
                    "num_candidates": len(sorted_candidates),
                },
            )
        ]

    def _build_trajectories(self, pairs: list[PreferencePair]) -> list[Trajectory]:
        """
        Convert preference pairs to Trajectory format with DPOItem episodes.

        Each trajectory contains:
        - episode: [DPOItem] with prompt, chosen_response, rejected_response
        - reward: score_diff (if reward_from_score_diff) or chosen_score
        - metadata: Contains pair information for tracking

        Args:
            pairs: List of preference pairs.

        Returns:
            List of trajectories with DPOItem episodes.
        """
        trajectories: list[Trajectory] = []

        for pair in pairs:
            # Create DPOItem from preference pair
            dpo_item = DPOItem(
                prompt=pair.prompt,
                chosen_response=pair.chosen_response,
                rejected_response=pair.rejected_response,
            )

            # Compute reward
            if self.config.reward_from_score_diff:
                reward = pair.score_diff
            else:
                reward = pair.chosen_score

            # Build trajectory with DPOItem episode
            trajectory = Trajectory(
                episode=[dpo_item],
                reward=reward,
                shaped_rewards=None,
                metadata={
                    # DPO-specific fields
                    "dpo_type": "preference_pair",
                    "score_diff": pair.score_diff,  # Tracking fields
                    "example_id": pair.example_id,
                    "turn_id": pair.turn_id,
                    "chosen_score": pair.chosen_score,
                    "rejected_score": pair.rejected_score,
                    "chosen_index": pair.chosen_index,
                    "rejected_index": pair.rejected_index,  # Additional metadata
                    **pair.metadata,
                },
            )

            trajectories.append(trajectory)

        return trajectories

    def _group_by_example(self, trajectories: list[Trajectory]) -> list[list[Trajectory]]:
        """
        Group trajectories by example ID for curriculum learning.

        This grouping enables:
        - Filtering by average reward per example
        - Expansion from easy to hard examples

        Args:
            trajectories: List of trajectories to group.

        Returns:
            List of trajectory lists, where each inner list contains
            trajectories for one example.
        """
        by_example: dict[str, list[Trajectory]] = {}

        for traj in trajectories:
            example_id = traj.metadata.get("example_id", "unknown")
            if example_id not in by_example:
                by_example[example_id] = []
            by_example[example_id].append(traj)

        return list(by_example.values())
