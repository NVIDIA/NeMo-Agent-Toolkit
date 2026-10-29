# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""`nat optimize` warns when an evaluator failed on items, since failures lower the objective silently."""

import logging
from types import SimpleNamespace

from nat.plugins.config_optimizer import eval_runtime_loader
from nat.plugins.config_optimizer.eval_runtime_loader import warn_on_evaluator_errors
from nat.plugins.eval.data_models.evaluator_io import EvalOutput
from nat.plugins.eval.data_models.evaluator_io import EvalOutputItem


def _accuracy_with_one_failure() -> EvalOutput:
    items = [EvalOutputItem(id=i, score=1.0, reasoning={}) for i in range(3)]
    items.append(EvalOutputItem(id=3, score=0.0, reasoning={"error": "judge timed out"}, error="judge timed out"))
    return EvalOutput(average_score=0.75, eval_output_items=items)


def test_warns_once_per_failing_evaluator(caplog):
    clean = EvalOutput(average_score=1.0, eval_output_items=[EvalOutputItem(id=1, score=1.0, reasoning={})])

    with caplog.at_level(logging.WARNING, logger=eval_runtime_loader.__name__):
        warn_on_evaluator_errors([("Accuracy", _accuracy_with_one_failure()), ("Relevance", clean)], context="trial 7")

    assert len(caplog.records) == 1
    assert caplog.records[0].getMessage() == ("trial 7: evaluator 'Accuracy' failed on 1/4 items; "
                                              "its average_score is lowered by the failures")


def test_no_warning_without_failures_or_items(caplog):
    clean = EvalOutput(average_score=1.0, eval_output_items=[EvalOutputItem(id=1, score=1.0, reasoning={})])

    with caplog.at_level(logging.WARNING, logger=eval_runtime_loader.__name__):
        warn_on_evaluator_errors([("Accuracy", clean), ("Latency", SimpleNamespace(average_score=0.5))],
                                 context="trial 0")

    assert caplog.records == []
