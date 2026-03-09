# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import logging
import typing
from collections.abc import Sequence
from typing import Literal

from tqdm import tqdm

from nat.data_models.atif import ATIFObservationResult
from nat.data_models.atif import ATIFStep
from nat.data_models.atif import ATIFTrajectory
from nat.data_models.evaluator import EvalOutput
from nat.plugins.eval.evaluator.atif_evaluator import AtifEvalSample
from nat.plugins.eval.evaluator.atif_evaluator import AtifEvalSampleList
from nat.plugins.eval.utils.tqdm_position_registry import TqdmPositionRegistry
from nat.utils.atif_message_utils import message_to_text
from nat.utils.atif_message_utils import trajectory_to_user_input

from .evaluate import _ragas_results_to_eval_output

if typing.TYPE_CHECKING:
    from ragas import EvaluationDataset
    from ragas.llms import LangchainLLMWrapper
    from ragas.messages import AIMessage
    from ragas.messages import HumanMessage
    from ragas.messages import ToolMessage
    from ragas.metrics import Metric

logger = logging.getLogger(__name__)

SampleType = Literal["single_turn", "multi_turn"]


def _observation_result_to_text(result: ATIFObservationResult) -> str:
    return message_to_text(result.content)


def _trajectory_to_retrieved_contexts(trajectory: ATIFTrajectory) -> list[str]:
    contexts: list[str] = []
    for step in trajectory.steps:
        if not step.observation:
            continue
        for result in step.observation.results:
            text = _observation_result_to_text(result)
            if text:
                contexts.append(text)
    return contexts


def _atif_step_to_ragas_messages(
    step: ATIFStep,
) -> list["HumanMessage | AIMessage | ToolMessage"]:
    """Convert a single ATIF step into one or more RAGAS message objects.

    Mapping:
        source="user"  → HumanMessage
        source="agent" → AIMessage (with optional ToolCall list)
                         followed by ToolMessage per observation result
        source="system" → HumanMessage (best-effort; RAGAS has no SystemMessage)
    """
    from ragas.messages import AIMessage as RagasAIMessage
    from ragas.messages import HumanMessage as RagasHumanMessage
    from ragas.messages import ToolCall as RagasToolCall
    from ragas.messages import ToolMessage as RagasToolMessage

    messages: list[RagasHumanMessage | RagasAIMessage | RagasToolMessage] = []
    text = message_to_text(step.message)

    if step.source == "user":
        messages.append(RagasHumanMessage(content=text))
        return messages

    if step.source == "system":
        messages.append(RagasHumanMessage(content=text))
        return messages

    ragas_tool_calls = []
    if step.tool_calls:
        for tc in step.tool_calls:
            ragas_tool_calls.append(
                RagasToolCall(name=tc.function_name, args=tc.arguments)
            )

    messages.append(RagasAIMessage(content=text, tool_calls=ragas_tool_calls or None))

    if step.observation and step.observation.results:
        for result in step.observation.results:
            tool_content = _observation_result_to_text(result)
            if tool_content:
                messages.append(RagasToolMessage(content=tool_content))

    return messages


def _atif_trajectory_to_multi_turn_messages(
    trajectory: ATIFTrajectory,
) -> list["HumanMessage | AIMessage | ToolMessage"]:
    """Convert an entire ATIF trajectory into a RAGAS multi-turn message sequence."""
    messages: list = []
    for step in trajectory.steps:
        messages.extend(_atif_step_to_ragas_messages(step))
    return messages


class RAGAtifEvaluator:

    def __init__(
        self,
        evaluator_llm: "LangchainLLMWrapper",
        metrics: Sequence["Metric"],
        max_concurrency: int = 8,
        sample_type: SampleType = "single_turn",
    ):
        self.evaluator_llm = evaluator_llm
        self.metrics = metrics
        self.max_concurrency = max_concurrency
        self.sample_type = sample_type

    def _build_single_turn_sample(self, sample: AtifEvalSample):
        """Build a RAGAS SingleTurnSample from an ATIF eval sample."""
        from ragas import SingleTurnSample

        user_input = trajectory_to_user_input(sample.trajectory)
        reference = sample.expected_output_obj
        response = sample.output_obj
        reference_contexts = [""]
        retrieved_contexts = _trajectory_to_retrieved_contexts(sample.trajectory)
        return SingleTurnSample(
            user_input=user_input,
            reference=reference,
            response=response,
            reference_contexts=reference_contexts,
            retrieved_contexts=retrieved_contexts,
        )

    def _build_multi_turn_sample(self, sample: AtifEvalSample):
        """Build a RAGAS MultiTurnSample from an ATIF eval sample."""
        from ragas import MultiTurnSample

        conversation = _atif_trajectory_to_multi_turn_messages(sample.trajectory)
        reference = sample.expected_output_obj
        kwargs: dict[str, typing.Any] = {"user_input": conversation}
        if reference is not None:
            kwargs["reference"] = str(reference)

        reference_tool_calls = sample.metadata.get("reference_tool_calls")
        if reference_tool_calls is not None:
            kwargs["reference_tool_calls"] = reference_tool_calls

        return MultiTurnSample(**kwargs)

    def atif_samples_to_ragas(self, atif_samples: AtifEvalSampleList) -> "EvaluationDataset":
        """Converts ATIF-native samples into a Ragas-compatible EvaluationDataset.

        Uses SingleTurnSample or MultiTurnSample depending on ``self.sample_type``.
        """
        from ragas import EvaluationDataset

        samples = []
        for sample in atif_samples:
            if self.sample_type == "multi_turn":
                ragas_sample = self._build_multi_turn_sample(sample)
            else:
                ragas_sample = self._build_single_turn_sample(sample)
            samples.append(ragas_sample)
        return EvaluationDataset(samples=samples)

    async def evaluate(self, atif_samples: AtifEvalSampleList) -> EvalOutput:
        """Run Ragas metrics evaluation on ATIF-native samples."""
        from ragas import evaluate as ragas_evaluate
        from ragas.run_config import RunConfig

        ragas_dataset = self.atif_samples_to_ragas(atif_samples)
        tqdm_position = TqdmPositionRegistry.claim()
        first_metric_name = self.metrics[0].name if self.metrics else "no-metrics"
        pbar = tqdm(total=len(ragas_dataset), desc=f"Evaluating Ragas {first_metric_name}", position=tqdm_position)
        try:
            if not self.metrics:
                logger.warning("No RAGAS metrics configured for ATIF evaluator; returning empty metric results.")
                results_dataset = None
            else:
                results_dataset = ragas_evaluate(dataset=ragas_dataset,
                                                 metrics=self.metrics,
                                                 show_progress=True,
                                                 llm=self.evaluator_llm,
                                                 run_config=RunConfig(max_workers=self.max_concurrency),
                                                 _pbar=pbar)
        except Exception:
            logger.exception("Error evaluating ATIF ragas metric")
            results_dataset = None
        finally:
            pbar.close()
            TqdmPositionRegistry.release(tqdm_position)

        ids = [sample.item_id for sample in atif_samples]
        return _ragas_results_to_eval_output(results_dataset=results_dataset, ids=ids)
