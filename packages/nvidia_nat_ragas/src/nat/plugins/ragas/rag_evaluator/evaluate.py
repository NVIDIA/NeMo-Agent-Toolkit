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

from pydantic import BaseModel

from nat.data_models.evaluator import EvalInputItem
from nat.data_models.evaluator import EvalOutputItem
from nat.data_models.intermediate_step import IntermediateStepType
from nat.plugins.eval.evaluator.base_evaluator import BaseEvaluator
from ragas import SingleTurnSample
from ragas.metrics.base import SimpleBaseMetric

from .data_models import EvalOutputItemRagasReasoning
from .utils import extract_metric_score
from .utils import nan_to_zero
from .utils import score_metric_result

logger = logging.getLogger(__name__)


class RAGEvaluator(BaseEvaluator):

    def __init__(self, metric: SimpleBaseMetric, max_concurrency: int = 8, input_obj_field: str | None = None):
        """Initialize evaluator with a single RAGAS metric."""
        metric_name = metric.name
        super().__init__(max_concurrency=max_concurrency, tqdm_desc=f"Evaluating Ragas {metric_name}")
        self.metric = metric
        self.input_obj_field = input_obj_field

    def _extract_input_obj(self, item: EvalInputItem) -> str:
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

    def _eval_input_item_to_ragas(self, item: EvalInputItem):
        """Convert one `EvalInputItem` into a ragas `SingleTurnSample`."""
        from nat.plugins.eval.utils.intermediate_step_adapter import IntermediateStepAdapter

        event_filter = [IntermediateStepType.TOOL_END, IntermediateStepType.LLM_END, IntermediateStepType.CUSTOM_END]
        intermediate_step_adapter = IntermediateStepAdapter()

        user_input = self._extract_input_obj(item)
        reference = item.expected_output_obj
        response = item.output_obj
        reference_contexts = [""]
        retrieved_contexts = intermediate_step_adapter.get_context(item.trajectory, event_filter)

        return SingleTurnSample(user_input=user_input,
                                reference=reference,
                                response=response,
                                reference_contexts=reference_contexts,
                                retrieved_contexts=retrieved_contexts)

    async def evaluate_item(self, item: EvalInputItem) -> EvalOutputItem:
        """Run configured ragas metric for one eval item and return one output item."""
        ragas_sample = self._eval_input_item_to_ragas(item)
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
            id=item.id,
            score=score,
            reasoning=reasoning.model_dump(exclude_none=True),
        )
