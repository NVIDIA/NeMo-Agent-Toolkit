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

from nat.data_models.atif import ATIFObservationResult
from nat.data_models.atif import ATIFTrajectory
from nat.data_models.evaluator import EvalOutput
from nat.data_models.evaluator import EvalOutputItem
from nat.plugins.eval.evaluator.atif_base_evaluator import AtifBaseEvaluator
from nat.plugins.eval.evaluator.atif_evaluator import AtifEvalSample
from nat.plugins.eval.evaluator.atif_evaluator import AtifEvalSampleList
from nat.utils.atif_message_utils import message_to_text
from nat.utils.atif_message_utils import trajectory_to_user_input

from .utils import nan_to_zero
from .utils import score_metric

if typing.TYPE_CHECKING:
    from ragas import EvaluationDataset
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import Metric

logger = logging.getLogger(__name__)

# Backward-compatible alias for existing imports/tests.
_score_metric = score_metric


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


class RAGAtifEvaluator(AtifBaseEvaluator):

    def __init__(self, evaluator_llm: "LangchainLLMWrapper", metrics: Sequence["Metric"], max_concurrency=8):
        super().__init__(max_concurrency=max_concurrency)
        self.evaluator_llm = evaluator_llm
        self.metrics = metrics

    def atif_sample_to_ragas(self, sample: AtifEvalSample):
        """Converts one ATIF sample into a ragas `SingleTurnSample`."""
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

    def atif_samples_to_ragas(self, atif_samples: AtifEvalSampleList) -> "EvaluationDataset":
        """Converts ATIF-native samples into a Ragas-compatible EvaluationDataset."""
        from ragas import EvaluationDataset

        samples = [self.atif_sample_to_ragas(sample) for sample in atif_samples]
        return EvaluationDataset(samples=samples)

    async def evaluate_atif_item(self, sample: AtifEvalSample) -> EvalOutputItem:
        """Run configured ragas metric for one ATIF sample and return one output item."""
        if not self.metrics:
            raise ValueError("No RAGAS metrics configured.")

        metric = self.metrics[0]
        ragas_sample = self.atif_sample_to_ragas(sample)
        raw_score = await _score_metric(metric, ragas_sample)
        score = nan_to_zero(raw_score)
        return EvalOutputItem(
            id=sample.item_id,
            score=score,
            reasoning={
                "user_input": ragas_sample.user_input,
                "reference": ragas_sample.reference,
                "response": ragas_sample.response,
                "retrieved_contexts": ragas_sample.retrieved_contexts,
            },
        )

    async def evaluate(self, atif_samples: AtifEvalSampleList) -> EvalOutput:
        """Run Ragas metrics evaluation on ATIF-native samples."""
        if not self.metrics:
            logger.warning("No RAGAS metrics configured for ATIF evaluator; returning empty metric results.")
            return EvalOutput(average_score=0.0, eval_output_items=[])
        return await self.evaluate_atif_fn(atif_samples)
