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

from nat.data_models.atif import ATIFObservationResult
from nat.data_models.atif import ATIFTrajectory
from nat.data_models.evaluator import EvalOutputItem
from nat.plugins.eval.evaluator.atif_base_evaluator import AtifBaseEvaluator
from nat.plugins.eval.evaluator.atif_evaluator import AtifEvalSample
from nat.utils.atif_message_utils import message_to_text
from nat.utils.atif_message_utils import trajectory_to_user_input
from ragas import SingleTurnSample
from ragas.metrics.base import SimpleBaseMetric

from .data_models import EvalOutputItemRagasReasoning
from .utils import extract_metric_score
from .utils import nan_to_zero
from .utils import score_metric_result


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

    def __init__(self, metric: SimpleBaseMetric, max_concurrency: int = 8):
        super().__init__(max_concurrency=max_concurrency)
        self.metric = metric

    @staticmethod
    def _atif_sample_to_ragas(sample: AtifEvalSample) -> SingleTurnSample:
        """Converts one ATIF sample into a ragas `SingleTurnSample`."""
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

    async def evaluate_atif_item(self, sample: AtifEvalSample) -> EvalOutputItem:
        """Run configured ragas metric for one ATIF sample and return one output item."""
        ragas_sample = self._atif_sample_to_ragas(sample)
        metric_result = await score_metric_result(self.metric, ragas_sample)
        raw_score = extract_metric_score(metric_result)
        score = nan_to_zero(raw_score)
        # stash the input and the ragas reasoning for analysis later
        reasoning = EvalOutputItemRagasReasoning(
            user_input=ragas_sample.user_input,
            reference=ragas_sample.reference,
            response=ragas_sample.response,
            retrieved_contexts=ragas_sample.retrieved_contexts,
            ragas_reason=metric_result.reason,
            ragas_traces=metric_result.traces,
        )
        return EvalOutputItem(
            id=sample.item_id,
            score=score,
            reasoning=reasoning.model_dump(exclude_none=True),
        )
