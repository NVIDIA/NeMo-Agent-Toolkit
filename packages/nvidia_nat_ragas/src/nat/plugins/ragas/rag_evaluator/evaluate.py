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

from pydantic import BaseModel

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
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import Metric

logger = logging.getLogger(__name__)


class RAGEvaluator(BaseEvaluator):

    def __init__(self,
                 evaluator_llm: "LangchainLLMWrapper",
                 metrics: Sequence["Metric"],
                 max_concurrency=8,
                 input_obj_field: str | None = None):
        metric_name = metrics[0].name if metrics else "no-metrics"
        super().__init__(max_concurrency=max_concurrency, tqdm_desc=f"Evaluating Ragas {metric_name}")
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

    async def evaluate_item(self, item: EvalInputItem) -> EvalOutputItem:
        """Run configured ragas metric for one eval item and return one output item."""
        if not self.metrics:
            raise ValueError("No RAGAS metrics configured.")

        metric = self.metrics[0]
        sample = self.eval_input_item_to_ragas(item)
        raw_score = await score_metric(metric, sample)
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
