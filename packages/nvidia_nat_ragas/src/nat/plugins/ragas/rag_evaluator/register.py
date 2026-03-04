# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from nat.data_models.evaluator import EvalInput
from nat.data_models.evaluator import EvalOutput
from nat.data_models.evaluator import EvaluatorLLMConfig
from nat.plugins.eval.evaluator.atif_evaluator import AtifEvalSampleList
from nat.utils.exception_handlers.automatic_retries import patch_with_retry

logger = logging.getLogger(__name__)


class RagasMetricConfig(BaseModel):
    """RAGAS metrics configuration."""

    skip: bool = False
    kwargs: dict | None = None


class RagasEvaluatorConfig(EvaluatorLLMConfig, name="ragas"):
    """Evaluation using RAGAS metrics."""

    metric: str | dict[str, RagasMetricConfig] = Field(
        default="AnswerAccuracy",
        description="RAGAS metric callable with optional 'kwargs:'",
    )
    input_obj_field: str | None = Field(
        default=None,
        description="The field in the input object that contains the content to evaluate.",
    )
    enable_atif_evaluator: bool = Field(
        default=False,
        description="Enable ATIF-native RAGAS evaluator lane. Disabled by default until rollout stabilization.",
    )

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
        """Returns metric configuration with defaults."""
        if isinstance(self.metric, str):
            return RagasMetricConfig()
        if isinstance(self.metric, dict) and self.metric:
            return next(iter(self.metric.values()))
        return RagasMetricConfig()


@register_evaluator(config_type=RagasEvaluatorConfig)
async def register_ragas_evaluator(config: RagasEvaluatorConfig, builder: EvalBuilder):
    from ragas.metrics import Metric

    def get_ragas_metric(metric_name: str) -> Metric | None:
        """Fetch callable for RAGAS metric by name."""
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
        """Run RAGAS evaluation and return NAT eval output."""
        if not evaluator:
            logger.warning("No evaluator found for RAGAS metrics.")
            return EvalOutput(average_score=0.0, eval_output_items=[])
        return await evaluator.evaluate(eval_input)

    async def evaluate_atif_fn(atif_samples: AtifEvalSampleList) -> EvalOutput:
        """Run ATIF-native RAGAS evaluation and return NAT eval output."""
        if not atif_evaluator:
            logger.warning("No ATIF evaluator found for RAGAS metrics.")
            return EvalOutput(average_score=0.0, eval_output_items=[])
        return await atif_evaluator.evaluate(atif_samples)

    from .evaluate import RAGEvaluator
    from .atif_evaluate import RAGAtifEvaluator

    llm = await builder.get_llm(config.llm_name, wrapper_type=LLMFrameworkEnum.LANGCHAIN)
    if config.do_auto_retry:
        llm = patch_with_retry(
            llm,
            retries=config.num_retries,
            retry_codes=config.retry_on_status_codes,
            retry_on_messages=config.retry_on_errors,
        )

    metrics = []
    metric_name = config.metric_name
    metric_config = config.metric_config
    if not metric_config.skip:
        metric_callable = get_ragas_metric(metric_name)
        if metric_callable:
            kwargs = metric_config.kwargs or {}
            metrics.append(metric_callable(**kwargs))

    evaluator = RAGEvaluator(evaluator_llm=llm,
                             metrics=metrics,
                             max_concurrency=builder.get_max_concurrency(),
                             input_obj_field=config.input_obj_field) if metrics else None
    atif_evaluator = RAGAtifEvaluator(evaluator_llm=llm,
                                      metrics=metrics,
                                      max_concurrency=builder.get_max_concurrency()) if (metrics and
                                                                                          config.enable_atif_evaluator) else None

    evaluator_info = EvaluatorInfo(config=config, evaluate_fn=evaluate_fn, description="Evaluator for RAGAS metrics")
    if config.enable_atif_evaluator:
        evaluator_info.evaluate_atif_fn = evaluate_atif_fn
    yield evaluator_info
