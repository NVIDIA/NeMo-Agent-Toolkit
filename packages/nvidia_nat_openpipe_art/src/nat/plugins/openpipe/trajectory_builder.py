# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
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

import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from nat.data_models.finetuning import EpisodeItem
from nat.data_models.finetuning import EpisodeItemRole
from nat.data_models.finetuning import Trajectory
from nat.data_models.finetuning import TrajectoryCollection
from nat.data_models.intermediate_step import IntermediateStep
from nat.data_models.intermediate_step import IntermediateStepCategory
from nat.eval.config import EvaluationRunOutput
from nat.eval.evaluator.evaluator_model import EvalInputItem
from nat.finetuning.interfaces.trajectory_builder import TrajectoryBuilder
from nat.finetuning.utils.parsers.base_parser import parse_to_openai_messages

from .config import ARTTrajectoryBuilderConfig

logger = logging.getLogger(__name__)


class ARTTrajectoryBuilder(TrajectoryBuilder):
    """
    Trajectory builder for the ART backend.
    """

    def __init__(
        self,
        trajectory_builder_config: ARTTrajectoryBuilderConfig,
    ):
        super().__init__(trajectory_builder_config=trajectory_builder_config)
        self.evaluation_runs: dict[str, list[asyncio.Task[EvaluationRunOutput]]] = {}

    @property
    def num_generations(self) -> int:
        return self.trajectory_builder_config.num_generations

    async def start_run(self, run_id: str, meta: dict | None = None) -> None:
        """
        Start multiple evaluation runs to collect trajectories.

        Args:
            run_id (str): The ID of the run.
            meta (dict): Metadata for the run.
        """

        if run_id in self.evaluation_runs:
            raise ValueError(f"Run {run_id} is already in progress.")

        logger.info("Starting %d evaluation runs for run_id: %s", self.num_generations, run_id)
        tasks = []

        for gen_idx in range(self.num_generations):
            task = asyncio.create_task(self.run_eval(), name=f"eval-run-{run_id}-gen-{gen_idx}")

            def _on_done(t: asyncio.Task[EvaluationRunOutput], generation_index: int = gen_idx) -> None:
                if t.cancelled():
                    logger.info("Evaluation run for run_id: %s, generation: %d was cancelled.",
                                run_id,
                                generation_index)
                elif exc := t.exception():
                    logger.error(
                        "Evaluation run for run_id: %s, generation: %d failed with exception: %s",
                        run_id,
                        generation_index,
                        exc,
                    )
                else:
                    logger.info(
                        "Evaluation run for run_id: %s, generation: %d completed successfully.",
                        run_id,
                        generation_index,
                    )

            task.add_done_callback(_on_done)
            tasks.append(task)

        self.evaluation_runs[run_id] = tasks

    async def finalize(self, run_id: str, meta: dict | None = None) -> TrajectoryCollection:
        """
        Waits for all evaluation runs to finalize and builds trajectories from
        the episode items, grouping them by example ID.

        Args:
            run_id (str): The ID of the run.
            meta (dict): Metadata for the run.

        Returns:
            TrajectoryCollection: The collection of built trajectories grouped by example.
        """
        from nat.eval.evaluator.evaluator_model import EvalOutputItem

        if run_id not in self.evaluation_runs:
            raise ValueError(f"No evaluation runs found for run_id: {run_id}")

        # Wait for all evaluation runs to complete
        tasks = self.evaluation_runs[run_id]
        eval_results = await asyncio.gather(*tasks)

        # Dictionary to group trajectories by example ID
        trajectories_by_id: dict[str, list[Trajectory]] = {}

        # Process each evaluation result
        for gen_idx, eval_result in enumerate(eval_results):
            reward_results: list[EvalOutputItem] | None = None
            for metric_name, metric_value in eval_result.evaluation_results:
                if metric_name == self.run_config.reward_function.name:
                    reward_results = metric_value.eval_output_items
                    break

            if not reward_results:
                logger.warning(f"No reward results found for run_id: {run_id}, generation: {gen_idx}")
                continue

            logger.info("Building trajectories for run_id: %s, generation: %d", run_id, gen_idx)

            # ---------- helpers ----------
            def _as_text(obj: Any) -> str:
                return (obj if isinstance(obj, str) else json.dumps(obj, ensure_ascii=False))

            def _parse_trajectory_from_steps(steps: list[IntermediateStep], ) -> list[EpisodeItem]:
                """Parse trajectory from intermediate steps using parser."""
                episode_items = []

                try:
                    # Use the base parser to convert to OpenAI messages
                    openai_messages = parse_to_openai_messages(steps)

                    # Convert OpenAI messages to EpisodeItems
                    for msg in openai_messages:
                        # Map OpenAI role to EpisodeItemRole
                        role_mapping = {
                            "user": EpisodeItemRole.USER,
                            "assistant": EpisodeItemRole.ASSISTANT,
                            "system": EpisodeItemRole.SYSTEM,
                            "tool": EpisodeItemRole.TOOL,
                            "function": EpisodeItemRole.FUNCTION,
                            "human": EpisodeItemRole.USER,
                            "ai": EpisodeItemRole.ASSISTANT,
                        }

                        role = role_mapping.get(msg.get("role"), EpisodeItemRole.OTHER)
                        content = msg.get("content", "")
                        logprobs = msg.get("logprobs")

                        # For assistant messages, skip if no logprobs
                        if role == EpisodeItemRole.ASSISTANT and not logprobs:
                            logger.debug("Skipping assistant message without logprobs")
                            continue

                        # Build metadata from message attributes
                        metadata = {}

                        # Add tool/function specific metadata
                        if "tool_call_id" in msg:
                            metadata["tool_call_id"] = msg["tool_call_id"]
                        if "tool_calls" in msg:
                            metadata["tool_calls"] = msg["tool_calls"]
                        if "function_call" in msg:
                            metadata["function_call"] = msg["function_call"]
                        if "name" in msg:
                            metadata["name"] = msg["name"]

                        episode_items.append(
                            EpisodeItem(
                                role=role,
                                content=content,
                                logprobs=logprobs,
                                metadata=metadata if metadata else None,
                            ))

                except ValueError as e:
                    logger.warning(
                        "Failed to parse trajectory using base parser: %s. "
                        "Falling back to empty episode.", str(e))
                    # Return empty list on parse failure
                    return []

                return episode_items

            # Create a mapping of id to input item for quick lookup
            input_items_map = {item.id: item for item in eval_result.eval_input.eval_input_items}

            for reward_item in reward_results:
                # Find the corresponding input item
                input_item: EvalInputItem = input_items_map.get(reward_item.id)
                if not input_item:
                    logger.warning(
                        "No input item found for reward item id: %s",
                        reward_item.id,
                    )
                    continue

                filtered_trajectory = []
                for item in input_item.trajectory:
                    if item.function_ancestry.function_name in self.run_config.target_functions:
                        # If target model is specified, filter by model name
                        if (self.run_config.target_model and item.event_category == IntermediateStepCategory.LLM
                                and item.payload.name != self.run_config.target_model):
                            continue
                        filtered_trajectory.append(item)

                if not filtered_trajectory:
                    logger.warning(
                        "No trajectory steps found for target function '%s' in item id: %s",
                        self.run_config.target_functions,
                        reward_item.id,
                    )
                    continue

                # Parse episode from intermediate steps
                episode = _parse_trajectory_from_steps(filtered_trajectory)

                # If no episode was parsed from steps, try to build from
                # input/output
                if not episode:
                    continue

                # Ensure we have at least a user and assistant message
                if len(episode) < 2:
                    logger.warning(
                        "Episode for item %s has less than 2 messages, skipping",
                        reward_item.id,
                    )
                    continue

                # Validate that assistant messages have logprobs
                # (required for training)
                has_valid_assistant = False
                for item in episode:
                    if item.role == EpisodeItemRole.ASSISTANT and item.logprobs:
                        has_valid_assistant = True
                        break

                if not has_valid_assistant:
                    logger.warning(
                        "Episode for item %s has no assistant messages with "
                        "logprobs, skipping as it cannot be used for training",
                        reward_item.id,
                    )
                    continue

                # Create trajectory
                trajectory = Trajectory(
                    episode=episode,
                    reward=(await self.compute_reward(reward_item, meta=meta)),
                    metadata={
                        "id": reward_item.id,
                        "reasoning": reward_item.reasoning,
                        "run_id": run_id,
                        "generation": gen_idx,
                    },
                )

                # Group by example ID
                if reward_item.id not in trajectories_by_id:
                    trajectories_by_id[reward_item.id] = []
                trajectories_by_id[reward_item.id].append(trajectory)

        # Clean up completed runs
        self.evaluation_runs.pop(run_id, None)

        # Convert dictionary to list of lists, maintaining order
        trajectories_list = list(trajectories_by_id.values())

        total_trajectories = sum(len(traj_list) for traj_list in trajectories_list)
        logger.info("Built %d trajectories across %d examples for run_id: %s",
                    total_trajectories,
                    len(trajectories_list),
                    run_id)

        # Flatten the trajectories list into a 1 d list of trajectories
        if not trajectories_list:
            logger.warning("No trajectories were built for run_id: %s", run_id)
            return TrajectoryCollection(trajectories=[], run_id=run_id)

        if self.num_generations == 1:
            # If only one generation, return flat list
            flat_trajectories = [traj for sublist in trajectories_list for traj in sublist]
            return TrajectoryCollection(trajectories=[flat_trajectories], run_id=run_id)

        return TrajectoryCollection(trajectories=trajectories_list, run_id=run_id)

    def log_progress(self, run_id: str, metrics: dict[str, Any], output_dir: str | None = None) -> None:
        """
        Log trajectory building progress.

        Args:
            run_id: The training run ID
            metrics: Dictionary of metrics to log
            output_dir: Optional output directory override
        """
        # Use default output directory if not provided
        out_dir = Path(output_dir) if output_dir else Path("./.tmp/nat/finetuning/trajectory_builder")
        out_dir.mkdir(parents=True, exist_ok=True)

        # Create log file for trajectory builder
        log_file = out_dir / f"trajectory_builder_{run_id}.jsonl"

        # Prepare log entry
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "run_id": run_id,
            "num_generations": self.num_generations,
            **metrics
        }

        # Append to log file
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(log_entry) + '\n')

        logger.debug("Trajectory builder progress logged for run %s: %d trajectories",
                     run_id,
                     metrics.get("num_trajectories", 0))
