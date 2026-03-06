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

from tqdm import tqdm

from nat.data_models.atif import ATIFObservationResult
from nat.data_models.atif import ATIFTrajectory
from nat.data_models.evaluator import EvalOutput
from nat.plugins.eval.evaluator.atif_evaluator import AtifEvalSampleList
from nat.plugins.eval.utils.tqdm_position_registry import TqdmPositionRegistry
from nat.utils.atif_message_utils import message_to_text
from nat.utils.atif_message_utils import trajectory_to_user_input

from .evaluate import _ragas_results_to_eval_output

if typing.TYPE_CHECKING:
    from ragas import EvaluationDataset
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import Metric

logger = logging.getLogger(__name__)


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


class RAGAtifEvaluator:

    def __init__(self, evaluator_llm: "LangchainLLMWrapper", metrics: Sequence["Metric"], max_concurrency=8):
        self.evaluator_llm = evaluator_llm
        self.metrics = metrics
        self.max_concurrency = max_concurrency

    def atif_samples_to_ragas(self, atif_samples: AtifEvalSampleList) -> "EvaluationDataset":
        """Converts ATIF-native samples into a Ragas-compatible EvaluationDataset."""
        from ragas import EvaluationDataset
        from ragas import SingleTurnSample

        samples = []
        for sample in atif_samples:
            user_input = trajectory_to_user_input(sample.trajectory)
            reference = sample.expected_output_obj
            response = sample.output_obj
            reference_contexts = [""]
            retrieved_contexts = _trajectory_to_retrieved_contexts(sample.trajectory)
            ragas_sample = SingleTurnSample(
                user_input=user_input,
                reference=reference,
                response=response,
                reference_contexts=reference_contexts,
                retrieved_contexts=retrieved_contexts,
            )
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
