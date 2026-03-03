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

from nat.cli.register_workflow import register_eval_callback
from nat.cli.register_workflow import register_optimizer_callback
from nat.plugins.opentelemetry.register import LangsmithTelemetryExporter


@register_eval_callback(config_type=LangsmithTelemetryExporter)
def _build_langsmith_eval_callback(config, **kwargs):
    from .langsmith_evaluation_callback import LangSmithEvaluationCallback

    return LangSmithEvaluationCallback(project=config.project)


@register_optimizer_callback(config_type=LangsmithTelemetryExporter)
def _build_langsmith_optimizer_callback(config, *, dataset_name=None, **kwargs):
    from .langsmith_optimization_callback import LangSmithOptimizationCallback

    return LangSmithOptimizationCallback(project=config.project, dataset_name=dataset_name)
