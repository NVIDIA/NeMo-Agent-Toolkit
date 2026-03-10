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
import math
import typing
from collections.abc import Sequence
from typing import Any

from pydantic import BaseModel
from tqdm import tqdm

from nat.data_models.evaluator import EvalInput
from nat.data_models.evaluator import EvalInputItem
from nat.data_models.evaluator import EvalOutput
from nat.data_models.evaluator import EvalOutputItem
from nat.data_models.intermediate_step import IntermediateStepType
from nat.plugins.eval.evaluator.base_evaluator import BaseEvaluator

from .utils import nan_to_zero
from .utils import score_metric

if typing.TYPE_CHECKING:
    # We are lazily importing ragas to avoid import-time side effects such as applying the nest_asyncio patch, which is
    # not compatible with Python 3.12+, we want to ensure that we are able to apply the nest_asyncio2 patch instead.
    from ragas import EvaluationDataset
    from ragas.dataset_schema import EvaluationResult
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import Metric

logger = logging.getLogger(__name__)

# Backward-compatible aliases for existing imports/tests.
_nan_to_zero = nan_to_zero
_score_metric = score_metric


def _ragas_results_to_eval_output(results_dataset: "EvaluationResult | None",
                                  ids: list[Any] | None = None) -> EvalOutput:
    """Convert a ragas EvaluationResult to NAT EvalOutput."""
    if not results_dataset:
        logger.error("Ragas evaluation failed with no results", exc_info=True)
        return EvalOutput(average_score=0.0, eval_output_items=[])

    scores: list[dict[str, float]] = results_dataset.scores
    if not scores:
        logger.warning("Ragas returned empty score list")
        return EvalOutput(average_score=0.0, eval_output_items=[])

    original_scores_dict = {metric: [score.get(metric) for score in scores] for metric in scores[0]}
    scores_dict = {metric: [nan_to_zero(score.get(metric)) for score in scores] for metric in scores[0]}
    first_metric_name = next(iter(scores_dict.keys()), None)

    average_scores = {metric: (sum(values) / len(values) if values else 0.0) for metric, values in scores_dict.items()}
    first_avg_score = average_scores.get(first_metric_name, 0.0)
    if isinstance(first_avg_score, float) and math.isnan(first_avg_score):
        first_avg_score = 0.0

    df = results_dataset.to_pandas()
    fallback_ids = df["user_input"].tolist()
    output_ids = ids if ids and len(ids) >= len(df) else fallback_ids

    eval_output_items = [
        EvalOutputItem(
            id=output_ids[i],
            score=original_scores_dict[first_metric_name][i] if first_metric_name else None,
            reasoning={
                key: getattr(row, key, None)
                for key in ["user_input", "reference", "response", "retrieved_contexts"]
            },
        ) for i, row in enumerate(df.itertuples(index=False))
    ]
    return EvalOutput(average_score=first_avg_score, eval_output_items=eval_output_items)


class RAGEvaluator(BaseEvaluator):

    def __init__(self,
                 evaluator_llm: "LangchainLLMWrapper",
                 metrics: Sequence["Metric"],
                 max_concurrency=8,
                 input_obj_field: str | None = None):
        metric_name = metrics[0].name if metrics else "no-metrics"
        super().__init__(max_concurrency=max_concurrency, tqdm_desc=f"Evaluating Ragas {metric_name}")
        self.evaluator_llm = evaluator_llm
        self.metrics = metrics
        self.input_obj_field = input_obj_field

    def extract_input_obj(self, item: EvalInputItem) -> str:
        """Extracts the input object from EvalInputItem based on the configured input_obj_field."""
        input_obj = item.input_obj
        if isinstance(input_obj, BaseModel):
            if self.input_obj_field and hasattr(input_obj, self.input_obj_field):
                # If input_obj_field is specified, return the value of that field
                return str(getattr(input_obj, self.input_obj_field, ""))

            # If no input_obj_field is specified, return the string representation of the model
            return input_obj.model_dump_json()

        if isinstance(input_obj, dict):
            # If input_obj is a dict, return the JSON string representation
            if self.input_obj_field and self.input_obj_field in input_obj:
                # If input_obj_field is specified, return the value of that field
                return str(input_obj[self.input_obj_field])

        return str(input_obj)  # Fallback to string representation of the dict

    def eval_input_item_to_ragas(self, item: EvalInputItem):
        """Convert one `EvalInputItem` into a ragas `SingleTurnSample`."""
        from nat.plugins.eval.utils.intermediate_step_adapter import IntermediateStepAdapter
        from ragas import SingleTurnSample

        event_filter = [IntermediateStepType.TOOL_END, IntermediateStepType.LLM_END, IntermediateStepType.CUSTOM_END]
        intermediate_step_adapter = IntermediateStepAdapter()

        user_input = self.extract_input_obj(item)
        reference = item.expected_output_obj
        response = item.output_obj
        reference_contexts = [""]
        retrieved_contexts = intermediate_step_adapter.get_context(item.trajectory, event_filter)

        return SingleTurnSample(user_input=user_input,
                                reference=reference,
                                response=response,
                                reference_contexts=reference_contexts,
                                retrieved_contexts=retrieved_contexts)

    def eval_input_to_ragas(self, eval_input: EvalInput) -> "EvaluationDataset":
        """Converts EvalInput into a Ragas-compatible EvaluationDataset."""
        from ragas import EvaluationDataset
        samples = [self.eval_input_item_to_ragas(item) for item in eval_input.eval_input_items]
        return EvaluationDataset(samples=samples)

    def ragas_to_eval_output(self, eval_input: EvalInput, results_dataset: "EvaluationResult | None") -> EvalOutput:
        """Converts the ragas EvaluationResult to nat EvalOutput"""
        ids = [item.id for item in eval_input.eval_input_items]
        return _ragas_results_to_eval_output(results_dataset=results_dataset, ids=ids)

    async def evaluate_item(self, item: EvalInputItem) -> EvalOutputItem:
        """Run configured ragas metric for one eval item and return one output item."""
        if not self.metrics:
            raise ValueError("No RAGAS metrics configured.")

        metric = self.metrics[0]
        sample = self.eval_input_item_to_ragas(item)
        raw_score = await _score_metric(metric, sample)
        score = nan_to_zero(raw_score)

        return EvalOutputItem(
            id=item.id,
            score=score,
            reasoning={
                "user_input": sample.user_input,
                "reference": sample.reference,
                "response": sample.response,
                "retrieved_contexts": sample.retrieved_contexts,
            },
        )

    async def evaluate(self, eval_input: EvalInput) -> EvalOutput:
        """Run Ragas metrics evaluation on the provided eval input."""
        if not self.metrics:
            logger.warning("No RAGAS metrics configured for evaluator; returning empty results.")
            return EvalOutput(average_score=0.0, eval_output_items=[])
        return await super().evaluate(eval_input)
