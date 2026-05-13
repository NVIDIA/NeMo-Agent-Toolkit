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

import asyncio
from types import SimpleNamespace

from nat.plugins.weave.register import _build_weave_eval_callback
from nat.plugins.weave.weave_eval_callback import WeaveEvaluationCallback


def test_register_builds_weave_eval_callback():
    config = SimpleNamespace(project="test-project")
    callback = _build_weave_eval_callback(config)
    assert isinstance(callback, WeaveEvaluationCallback)


def test_weave_eval_callback_noops_without_weave_runtime():
    callback = WeaveEvaluationCallback(project="test-project")

    callback.evaluation_logger_cls = None
    callback.weave_client_context = None

    with callback.evaluation_context():
        pass

    callback.on_eval_started(workflow_alias="wf",
                             eval_input=SimpleNamespace(eval_input_items=[]),
                             config=SimpleNamespace())
    callback.on_prediction(item=SimpleNamespace(id="1"), output={"text": "ok"})
    asyncio.run(
        callback.a_on_usage_stats(item=SimpleNamespace(id="1"),
                                  usage_stats_item=SimpleNamespace(runtime=1.0, total_tokens=5)))
    asyncio.run(callback.a_on_evaluator_score(eval_output=SimpleNamespace(eval_output_items=[]), evaluator_name="acc"))
    asyncio.run(callback.a_on_export_flush())
    callback.on_eval_summary(usage_stats=SimpleNamespace(total_runtime=1.0),
                             evaluation_results=[],
                             profiler_results=SimpleNamespace(llm_latency_ci=None, workflow_runtime_metrics=None))
