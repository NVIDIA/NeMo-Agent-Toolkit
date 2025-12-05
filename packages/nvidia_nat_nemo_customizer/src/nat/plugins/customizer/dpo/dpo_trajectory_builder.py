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
workflows that produce scored candidate intermediate steps (e.g., dpo_candidate_move
steps from the DPO Tic-Tac-Toe example).

The builder:
1. Runs evaluation to collect intermediate steps
2. Filters for CUSTOM steps with the configured name
3. Groups candidates by turn_id
4. Generates preference pairs based on score differences
5. Builds trajectories in NAT's format for DPO training
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

from nat.data_models.finetuning import EpisodeItem
from nat.data_models.finetuning import EpisodeItemRole
from nat.data_models.finetuning import Trajectory
from nat.data_models.finetuning import TrajectoryCollection
from nat.data_models.intermediate_step import IntermediateStep
from nat.data_models.intermediate_step import IntermediateStepCategory
from nat.data_models.intermediate_step import IntermediateStepType
from nat.eval.config import EvaluationRunOutput
from nat.eval.evaluator.evaluator_model import EvalInputItem
from nat.finetuning.interfaces.trajectory_builder import TrajectoryBuilder

from .config import DPOTrajectoryBuilderConfig

logger = logging.getLogger(__name__)

# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class CandidateStep:
    """
    Parsed candidate from an intermediate step.

    Represents a single candidate response that was generated and scored
    for a particular turn in the workflow.
    """

    example_id: str
    """Unique identifier for the dataset example."""

    turn_id: str
    """Identifier for the turn (groups candidates competing for the same prompt)."""

    candidate_index: int
    """Index of this candidate within the turn."""

    prompt: str
    """Input prompt that produced this response."""

    response: str
    """Model's response/completion."""

    score: float
    """Score assigned to this candidate (higher is better)."""

    is_selected: bool
    """Whether this candidate was selected as the final response."""

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

    prompt: str
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

    This builder collects preference pairs from workflows that produce scored
    candidate intermediate steps. It groups candidates by turn_id and creates
    preference pairs based on score differences.

    Key features:
    - Flexible field mapping for different workflow formats
    - Exhaustive or best-vs-worst pair generation modes
    - Configurable score difference filtering
    - Grouping by example for curriculum learning

    Example workflow integration:
    ```yaml
    trajectory_builders:
      dpo_builder:
        _type: dpo_traj_builder
        custom_step_name: dpo_candidate_move
        exhaustive_pairs: true
        min_score_diff: 0.05
    ```
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
            self.config.custom_step_name,
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
        Wait for evaluation, collect CUSTOM steps, and build DPO trajectories.

        This method:
        1. Waits for the evaluation run to complete
        2. Collects and groups candidates by turn_id
        3. Generates preference pairs
        4. Builds trajectories in NAT's format
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

        # Step 3: Build trajectories
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
        out_dir = Path(output_dir) if output_dir else Path("./.tmp/nat/finetuning/dpo_trajectory_builder")
        out_dir.mkdir(parents=True, exist_ok=True)

        # Create log file
        log_file = out_dir / f"dpo_trajectory_builder_{run_id}.jsonl"

        # Prepare log entry
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "run_id": run_id,
            "config": {
                "custom_step_name": self.config.custom_step_name,
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
        Extract CUSTOM intermediate steps and group by turn_id.

        This method:
        1. Iterates through all evaluation input items
        2. Filters for CUSTOM_END steps with the configured name
        3. Parses metadata into CandidateStep objects
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
            # Filter for CUSTOM_END steps with matching name
            for step in input_item.trajectory:
                if not self._is_target_step(step):
                    continue

                # Parse candidate from step metadata
                candidate = self._parse_candidate(example_id, step)
                if candidate is None:
                    continue

                self._metrics["total_candidates"] = self._metrics.get("total_candidates", 0) + 1

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
        Check if an intermediate step is a target CUSTOM step.

        Args:
            step: The intermediate step to check.

        Returns:
            True if this is a CUSTOM_END step with the configured name.
        """
        return (step.event_category == IntermediateStepCategory.CUSTOM
                and step.event_type == IntermediateStepType.CUSTOM_END
                and step.payload.name == self.config.custom_step_name)

    def _parse_candidate(self, example_id: str, step: IntermediateStep) -> CandidateStep | None:
        """
        Parse a CandidateStep from an intermediate step.

        Args:
            example_id: The example ID this step belongs to.
            step: The intermediate step to parse.

        Returns:
            CandidateStep if parsing succeeds, None otherwise.
        """
        metadata = step.payload.metadata
        if metadata is None:
            logger.warning("Step has no metadata, skipping: %s", step.payload.UUID)
            return None

        # Handle both dict and TraceMetadata types
        if hasattr(metadata, "model_dump"):
            metadata = metadata.model_dump()

        # Extract required fields using configured keys
        try:
            turn_id = metadata.get(self.config.turn_id_key)
            if turn_id is None:
                logger.warning(
                    "Step missing turn_id key '%s', skipping: %s",
                    self.config.turn_id_key,
                    step.payload.UUID,
                )
                return None

            score = metadata.get(self.config.score_key)
            if score is None:
                logger.warning(
                    "Step missing score key '%s', skipping: %s",
                    self.config.score_key,
                    step.payload.UUID,
                )
                return None

            prompt = metadata.get(self.config.prompt_key, "")
            response = metadata.get(self.config.response_key, "")
            candidate_index = metadata.get(self.config.candidate_index_key, 0)
            is_selected = metadata.get("is_selected", False)

            return CandidateStep(
                example_id=str(example_id),
                turn_id=str(turn_id),
                candidate_index=int(candidate_index),
                prompt=str(prompt),
                response=str(response),
                score=float(score),
                is_selected=bool(is_selected),
                raw_metadata=metadata,
            )

        except (KeyError, TypeError, ValueError) as e:
            logger.warning(
                "Failed to parse candidate from step %s: %s",
                step.payload.UUID,
                e,
            )
            return None

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
                            "chosen_is_selected": chosen.is_selected,
                            "rejected_is_selected": rejected.is_selected,
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
                    "chosen_is_selected": chosen.is_selected,
                    "rejected_is_selected": rejected.is_selected,
                    "num_candidates": len(sorted_candidates),
                },
            )
        ]

    def _build_trajectories(self, pairs: list[PreferencePair]) -> list[Trajectory]:
        """
        Convert preference pairs to Trajectory format.

        Each trajectory contains:
        - episode: [user_prompt, assistant_chosen_response]
        - reward: score_diff (if reward_from_score_diff) or chosen_score
        - metadata: Contains rejected_response and pair information

        The rejected response is stored in metadata because NAT's Trajectory
        format represents a single rollout. DPO training backends should extract
        the rejected response from metadata.

        Args:
            pairs: List of preference pairs.

        Returns:
            List of trajectories.
        """
        trajectories: list[Trajectory] = []

        for pair in pairs:
            # Build episode items
            episode: list[EpisodeItem] = []

            # Optionally add system prompt
            if self.config.include_system_prompt:
                system_prompt = pair.metadata.get(
                    self.config.system_prompt_key,
                    pair.metadata.get("chosen_raw_metadata", {}).get(self.config.system_prompt_key),
                )
                if system_prompt:
                    episode.append(
                        EpisodeItem(
                            role=EpisodeItemRole.SYSTEM,
                            content=str(system_prompt),
                            logprobs=None,
                            metadata=None,
                        ))

            # Add user prompt
            episode.append(EpisodeItem(
                role=EpisodeItemRole.USER,
                content=pair.prompt,
                logprobs=None,
                metadata=None,
            ))

            # Add chosen assistant response
            # Note: We use an empty dict for logprobs to satisfy EpisodeItem validation
            # DPO training doesn't require logprobs from the data (it computes them)
            episode.append(
                EpisodeItem(
                    role=EpisodeItemRole.ASSISTANT,
                    content=pair.chosen_response,
                    logprobs={},  # Empty dict satisfies validation
                    metadata={
                        "dpo_chosen": True,
                        "score": pair.chosen_score,
                        "candidate_index": pair.chosen_index,
                    },
                ))

            # Compute reward
            if self.config.reward_from_score_diff:
                reward = pair.score_diff
            else:
                reward = pair.chosen_score

            # Build trajectory with rejected response in metadata
            trajectory = Trajectory(
                episode=episode,
                reward=reward,
                shaped_rewards=None,
                metadata={
                    # DPO-specific fields
                    "dpo_type": "preference_pair",
                    "rejected_response": pair.rejected_response,
                    "rejected_score": pair.rejected_score,
                    "rejected_index": pair.rejected_index,
                    "score_diff": pair.score_diff,  # Tracking fields
                    "example_id": pair.example_id,
                    "turn_id": pair.turn_id,
                    "prompt": pair.prompt,
                    "chosen_score": pair.chosen_score,
                    "chosen_index": pair.chosen_index,  # Additional metadata
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
