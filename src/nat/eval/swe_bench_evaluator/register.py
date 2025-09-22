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

from pydantic import Field

from nat.builder.builder import EvalBuilder
from nat.builder.evaluator import EvaluatorInfo
from nat.cli.register_workflow import register_evaluator
from nat.data_models.evaluator import EvaluatorBaseConfig


class SweBenchEvaluatorConfig(EvaluatorBaseConfig, name="swe_bench"):
    """Code patch evaluation for SWE Bench problems."""

    run_id: str = Field(description="swe-bench test harness run identifier.")


@register_evaluator(config_type=SweBenchEvaluatorConfig)
async def register_swe_bench_evaluator(config: SweBenchEvaluatorConfig, builder: EvalBuilder):

    from .evaluate import SweBenchEvaluator
    _evaluator = SweBenchEvaluator(config.run_id, builder.get_max_concurrency(), builder.get_output_dir())

    yield EvaluatorInfo(config=config, evaluate_fn=_evaluator.evaluate, description="SWE Bench Evaluator")
