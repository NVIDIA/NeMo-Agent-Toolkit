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

from pydantic import BaseModel
from pydantic import Field
from pydantic import model_validator

from nat.builder.builder import EvalBuilder
from nat.builder.evaluator import EvaluatorInfo
from nat.builder.framework_enum import LLMFrameworkEnum
from nat.cli.register_workflow import register_evaluator
from nat.data_models.evaluator import EvaluatorBaseConfig
from nat.eval.evaluator.evaluator_model import EvalInput
from nat.eval.evaluator.evaluator_model import EvalOutput

logger = logging.getLogger(__name__)


class RagasMetricConfig(BaseModel):
    ''' RAGAS metrics configuration
    skip: Allows the metric config to be present but not used
    kwargs: Additional arguments to pass to the metric's callable
    '''
    skip: bool = False
    # kwargs specific to the metric's callable
    kwargs: dict | None = None


class RagasEvaluatorConfig(EvaluatorBaseConfig, name="ragas"):
    """Evaluation using RAGAS metrics."""

    llm_name: str = Field(description="LLM as a judge.")
    # Ragas metric
    metric: str | dict[str, RagasMetricConfig] = Field(default="AnswerAccuracy",
                                                       description="RAGAS metric callable with optional 'kwargs:'")
    input_obj_field: str | None = Field(
        default=None, description="The field in the input object that contains the content to evaluate.")

    @model_validator(mode="before")
    @classmethod
    def validate_metric(cls, values):
        """Ensures metric is either a string or a single-item dictionary."""
        metric = values.get("metric")

        if isinstance(metric, dict):
            if len(metric) != 1:
                raise ValueError("Only one metric is allowed in the configuration.")
            _, value = next(iter(metric.items()))
            if not isinstance(value, dict):
                raise ValueError("Metric value must be a RagasMetricConfig object.")
        elif not isinstance(metric, str):
            raise ValueError("Metric must be either a string or a single-item dictionary.")

        return values

    @property
    def metric_name(self) -> str:
        """Returns the single metric name."""
        if isinstance(self.metric, str):
            return self.metric
        if isinstance(self.metric, dict) and self.metric:
            return next(iter(self.metric.keys()))
        return ""

    @property
    def metric_config(self) -> RagasMetricConfig:
        """Returns the metric configuration (or a default if only a string is provided)."""
        if isinstance(self.metric, str):
            return RagasMetricConfig()  # Default config when only a metric name is given
        if isinstance(self.metric, dict) and self.metric:
            return next(iter(self.metric.values()))
        return RagasMetricConfig()  # Default config when an invalid type is provided


@register_evaluator(config_type=RagasEvaluatorConfig)
async def register_ragas_evaluator(config: RagasEvaluatorConfig, builder: EvalBuilder):
    from ragas.metrics import Metric

    def get_ragas_metric(metric_name: str) -> Metric | None:
        """
        Fetch callable for RAGAS metrics
        """
        try:
            import ragas.metrics as ragas_metrics

            return getattr(ragas_metrics, metric_name)
        except ImportError as e:
            message = f"Ragas metrics not found {e}."
            logger.error(message)
            raise ValueError(message) from e
        except AttributeError as e:
            message = f"Ragas metric {metric_name} not found {e}."
            logger.exception(message)
            return None

    async def evaluate_fn(eval_input: EvalInput) -> EvalOutput:
        '''Run the RAGAS evaluation and return the average scores and evaluation results dataframe'''
        if not _evaluator:
            logger.warning("No evaluator found for RAGAS metrics.")
            # return empty results if no evaluator is found
            return EvalOutput(average_score=0.0, eval_output_items=[])

        return await _evaluator.evaluate(eval_input)

    from .evaluate import RAGEvaluator

    # Get LLM
    llm = await builder.get_llm(config.llm_name, wrapper_type=LLMFrameworkEnum.LANGCHAIN)

    # Get RAGAS metric callable from the metric config and create a list of metric-callables
    metrics = []
    # currently only one metric is supported
    metric_name = config.metric_name  # Extracts the metric name
    metric_config = config.metric_config  # Extracts the config (handles str/dict cases)

    # Skip if `skip` is True
    if not metric_config.skip:
        metric_callable = get_ragas_metric(metric_name)
        if metric_callable:
            kwargs = metric_config.kwargs or {}
            metrics.append(metric_callable(**kwargs))

    # Create the RAG evaluator
    _evaluator = RAGEvaluator(evaluator_llm=llm,
                              metrics=metrics,
                              max_concurrency=builder.get_max_concurrency(),
                              input_obj_field=config.input_obj_field) if metrics else None

    yield EvaluatorInfo(config=config, evaluate_fn=evaluate_fn, description="Evaluator for RAGAS metrics")
