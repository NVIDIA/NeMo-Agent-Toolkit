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

import logging

from nat.builder.builder import EvalBuilder
from nat.builder.evaluator import EvaluatorInfo
from nat.cli.register_workflow import register_evaluator
from nat.data_models.evaluator import EvaluatorBaseConfig
from nat.eval.evaluator.base_evaluator import BaseEvaluator
from nat.eval.evaluator.evaluator_model import EvalInputItem
from nat.eval.evaluator.evaluator_model import EvalOutputItem

logger = logging.getLogger(__name__)


class ClassificationEvaluatorConfig(EvaluatorBaseConfig, name="classification_accuracy"):
    """Configuration for custom classification evaluator.

    This evaluator config is used to evaluate the accuracy of classification predictions
    by comparing them against expected labels.
    """
    pass


@register_evaluator(config_type=ClassificationEvaluatorConfig)
async def register_classification_evaluator(config: ClassificationEvaluatorConfig, builder: EvalBuilder):
    """Register a custom classification evaluator.

    Args:
        config: Configuration object for the evaluator
        builder: EvalBuilder instance to access evaluation context

    Returns:
        EvaluatorInfo containing the evaluator configuration and evaluation function
    """
    evaluator = ClassificationEvaluator(builder.get_max_concurrency())

    yield EvaluatorInfo(config=config, evaluate_fn=evaluator.evaluate, description="Classification Accuracy Evaluator")


class ClassificationEvaluator(BaseEvaluator):

    def __init__(
        self,
        max_concurrency: int = 8,
    ):
        super().__init__(max_concurrency=max_concurrency, tqdm_desc="Evaluating classification accuracy")
        logger.debug("Classification accuracy evaluator initialized.")

    async def evaluate_item(self, item: EvalInputItem) -> EvalOutputItem:
        """Compute accuracy score for an individual prediction.

        Extracts the predicted category from the generated answer and compares
        it to the expected answer.

        Args:
            item: Single evaluation item containing prediction and ground truth

        Returns:
            EvalOutputItem containing the accuracy score and reasoning
        """
        label = item.full_dataset_entry['label']
        generated_answer = item.output_obj

        try:
            # Extract predicted category from generated answer
            prediction = generated_answer.split('Root Cause Category')[-1].strip().split('\n')[0].lower().strip()
            if prediction == label:
                score = 1.0
                reasoning = f"The prediction {prediction} is correct. (label: {label})"
            else:
                score = 0.0
                reasoning = f"The prediction {prediction} is incorrect. (label: {label})"
        except Exception:
            score = 0.0
            reasoning = f"The prediction is not in the expected format: {generated_answer}"

        return EvalOutputItem(id=item.id, score=score, reasoning=reasoning)
