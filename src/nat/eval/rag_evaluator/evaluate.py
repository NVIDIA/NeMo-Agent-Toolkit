# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from collections.abc import Sequence

from pydantic import BaseModel
from ragas import EvaluationDataset
from ragas import SingleTurnSample
from ragas.dataset_schema import EvaluationResult
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import Metric
from tqdm import tqdm

from nat.data_models.intermediate_step import IntermediateStepType
from nat.eval.evaluator.evaluator_model import EvalInput
from nat.eval.evaluator.evaluator_model import EvalInputItem
from nat.eval.evaluator.evaluator_model import EvalOutput
from nat.eval.evaluator.evaluator_model import EvalOutputItem
from nat.eval.utils.tqdm_position_registry import TqdmPositionRegistry

logger = logging.getLogger(__name__)


class RAGEvaluator:

    def __init__(self,
                 evaluator_llm: LangchainLLMWrapper,
                 metrics: Sequence[Metric],
                 max_concurrency=8,
                 input_obj_field: str | None = None):
        self.evaluator_llm = evaluator_llm
        self.metrics = metrics
        self.max_concurrency = max_concurrency
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

    def eval_input_to_ragas(self, eval_input: EvalInput) -> EvaluationDataset:
        """Converts EvalInput into a Ragas-compatible EvaluationDataset."""
        from nat.eval.intermediate_step_adapter import IntermediateStepAdapter
        event_filter = [IntermediateStepType.TOOL_END, IntermediateStepType.LLM_END, IntermediateStepType.CUSTOM_END]
        samples = []

        intermediate_step_adapter = IntermediateStepAdapter()
        for item in eval_input.eval_input_items:
            # Extract required fields from EvalInputItem
            user_input = self.extract_input_obj(item)  # Extract input object as string
            reference = item.expected_output_obj  # Reference correct answer
            response = item.output_obj  # Model's generated response

            # Handle context extraction from trajectory if available
            reference_contexts = [""]  # Default to empty context
            # implement context extraction from expected_trajectory

            retrieved_contexts = intermediate_step_adapter.get_context(item.trajectory, event_filter)
            # implement context extraction from expected_trajectory

            # Create a SingleTurnSample
            sample = SingleTurnSample(
                user_input=user_input,
                reference=reference,
                response=response,
                reference_contexts=reference_contexts,
                retrieved_contexts=retrieved_contexts,
            )
            samples.append(sample)

        return EvaluationDataset(samples=samples)

    def ragas_to_eval_output(self, eval_input: EvalInput, results_dataset: EvaluationResult | None) -> EvalOutput:
        """Converts the ragas EvaluationResult to aiq EvalOutput"""

        if not results_dataset:
            logger.error("Ragas evaluation failed with no results")
            return EvalOutput(average_score=0.0, eval_output_items=[])

        scores: list[dict[str, float]] = results_dataset.scores

        # If Ragas returned no scores, return empty output to avoid downstream errors
        if not scores:
            logger.warning("Ragas returned empty score list")
            return EvalOutput(average_score=0.0, eval_output_items=[])

        def _nan_to_zero(v: float | None) -> float:
            """Convert NaN or None to 0.0 for safe arithmetic/serialization."""
            return 0.0 if v is None or (isinstance(v, float) and math.isnan(v)) else v

        # Convert from list of dicts to dict of lists, coercing NaN/None to 0.0
        scores_dict = {metric: [_nan_to_zero(score.get(metric)) for score in scores] for metric in scores[0]}
        first_metric_name = list(scores_dict.keys())[0] if scores_dict else None

        # Compute the average of each metric, guarding against empty lists
        average_scores = {
            metric: (sum(values) / len(values) if values else 0.0)
            for metric, values in scores_dict.items()
        }

        first_avg_score = average_scores.get(list(scores_dict.keys())[0], 0.0)
        if isinstance(first_avg_score, float) and math.isnan(first_avg_score):
            first_avg_score = 0.0

        df = results_dataset.to_pandas()
        # Get id from eval_input if df size matches number of eval_input_items
        if len(eval_input.eval_input_items) >= len(df):
            ids = [item.id for item in eval_input.eval_input_items]  # Extract IDs
        else:
            ids = df["user_input"].tolist()  # Use "user_input" as ID fallback

        # Construct EvalOutputItem list
        eval_output_items = [
            EvalOutputItem(
                id=ids[i],
                score=_nan_to_zero(getattr(row, first_metric_name, 0.0) if first_metric_name else 0.0),
                reasoning={
                    key:
                        getattr(row, key, None)  # Use getattr to safely access attributes
                    for key in ["user_input", "reference", "response", "retrieved_contexts"]
                }) for i, row in enumerate(df.itertuples(index=False))
        ]
        # Return EvalOutput
        return EvalOutput(average_score=first_avg_score, eval_output_items=eval_output_items)

    async def evaluate(self, eval_input: EvalInput) -> EvalOutput:
        """Run Ragas metrics evaluation on the provided EvalInput"""
        from ragas import evaluate as ragas_evaluate
        from ragas.run_config import RunConfig

        ragas_dataset = self.eval_input_to_ragas(eval_input)
        tqdm_position = TqdmPositionRegistry.claim()
        first_metric_name = self.metrics[0].name
        pbar = tqdm(total=len(ragas_dataset), desc=f"Evaluating Ragas {first_metric_name}", position=tqdm_position)
        try:
            results_dataset = ragas_evaluate(dataset=ragas_dataset,
                                             metrics=self.metrics,
                                             show_progress=True,
                                             llm=self.evaluator_llm,
                                             run_config=RunConfig(max_workers=self.max_concurrency),
                                             _pbar=pbar)
        except Exception as e:
            # On exception we still continue with other evaluators. Log and return an avg_score of 0.0
            logger.exception("Error evaluating ragas metric, Error: %s", e, exc_info=True)
            results_dataset = None
        finally:
            pbar.close()
            TqdmPositionRegistry.release(tqdm_position)

        return self.ragas_to_eval_output(eval_input, results_dataset)
