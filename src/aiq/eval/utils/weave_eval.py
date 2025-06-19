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

import asyncio
import logging
from typing import Any
from typing import List

from aiq.eval.evaluator.evaluator_model import EvalInput
from aiq.eval.evaluator.evaluator_model import EvalInputItem
from aiq.eval.evaluator.evaluator_model import EvalOutput

logger = logging.getLogger(__name__)


class WeaveEvaluationIntegration:  # pylint: disable=too-many-public-methods
    """
    Class to handle all Weave integration functionality.
    """

    def __init__(self):
        self.available = False
        self.client = None
        self.eval_logger = None
        self.pred_loggers = {}

        try:
            from weave.flow.eval_imperative import EvaluationLogger
            from weave.flow.eval_imperative import ScoreLogger
            from weave.trace.context import weave_client_context
            self.EvaluationLogger = EvaluationLogger
            self.ScoreLogger = ScoreLogger
            self.weave_client_context = weave_client_context
            self.available = True
        except ImportError:
            self.available = False
            # we simply don't do anything if weave is not available
            pass

    def initialize_client(self):
        """Initialize the Weave client if available."""
        if not self.available:
            return False

        try:
            self.client = self.weave_client_context.require_weave_client()
            return self.client is not None
        except Exception:
            self.client = None
            return False

    def initialize_logger(self, eval_input: EvalInput, config: Any):
        """Initialize the Weave evaluation logger."""
        if not self.client:
            return False

        try:
            weave_dataset = [
                item.model_dump(exclude={"output_obj", "trajectory"}) for item in eval_input.eval_input_items
            ]
            config_dict = config.model_dump(mode="json")
            # TODO: make this configurable
            config_dict["name"] = "aiqtoolkit-eval"
            self.eval_logger = self.EvaluationLogger(model=config_dict, dataset=weave_dataset)
            self.pred_loggers = {}

            del weave_dataset
            del config_dict
            return True
        except Exception as e:
            self.eval_logger = None
            logger.warning("Failed to initialize Weave `EvaluationLogger`: %s", e)

            return False

    def log_prediction(self, item: EvalInputItem, output: Any):
        """Log a prediction to Weave."""
        if not self.eval_logger:
            return

        pred_logger = self.eval_logger.log_prediction(inputs=item.model_dump(exclude={"output_obj", "trajectory"}),
                                                      output=output)
        self.pred_loggers[item.id] = pred_logger

    async def alog_score(self, eval_output: EvalOutput, evaluator_name: str):
        """Log scores for evaluation outputs."""
        if not self.eval_logger:
            return

        for eval_output_item in eval_output.eval_output_items:
            if eval_output_item.id in self.pred_loggers:
                await self.pred_loggers[eval_output_item.id].alog_score(
                    scorer=evaluator_name,
                    score=eval_output_item.score,
                )

    async def afinish_loggers(self):
        """Finish all prediction loggers."""
        if not self.eval_logger:
            return

        async def _finish_one(pred_logger):
            if hasattr(pred_logger, '_has_finished') and not pred_logger._has_finished:
                return
            # run the *blocking* finish() in a thread so we don’t nest loops
            await asyncio.to_thread(pred_logger.finish)

        await asyncio.gather(*[_finish_one(pl) for pl in self.pred_loggers.values()])

    def log_summary(self, evaluation_results: List[tuple[str, EvalOutput]]):
        """Log summary statistics to Weave."""
        if not self.eval_logger:
            return

        summary = {}
        for evaluator_name, eval_output in evaluation_results:
            # Calculate average score for this evaluator
            scores = [item.score for item in eval_output.eval_output_items if item.score is not None]
            if scores:
                summary[f"{evaluator_name}_avg"] = sum(scores) / len(scores)

        # Log the summary to finish the evaluation
        self.eval_logger.log_summary(summary)
