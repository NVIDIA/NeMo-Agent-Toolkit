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
"""
This file defines the workflow for solving problems in the SWE Bench dataset.

Two types of predictors have been provided:
1. **gold**: Uses the patch from the input, bypassing problem-solving logic. See predictors/predict_gold_stub.py.
2. **full**: Full problem-solving workflow (TO BE IMPLEMENTED). See predictors/predict_full.py.

### Implementation Guide for the Full Predictor:
To implement the full predictor, populate the following functions in the predictors/full_predict.py file:
1. `workflow_base_fn`: Setup the prompt and agents needed by the workflow.
2. `predict_fn`: Implement the problem-solving logic for one swe-bench instance.

### You can add more predictors by following these steps:
1. Create a new file in the predictors directory.
2. Add a concrete class using the abstrach base class predictors.predict_abc.SweBenchPredictorBase.
3. Register the class with and unique name using the `@register_predictor` decorator.
4. Import the class in this file to populate the `PredictorRegistry`.
"""

import logging

# flake8: noqa: F401, pylint: disable=unused-import
from aiq_swe_bench import register_tools
from aiq_swe_bench.config import SweBenchWorkflowConfig

from aiq.builder.builder import Builder
from aiq.cli.register_workflow import register_function

logger = logging.getLogger(__name__)


@register_function(config_type=SweBenchWorkflowConfig)
async def swe_bench_workflow(config: SweBenchWorkflowConfig, builder: Builder):
    '''Workflow for solving SWE bench problems'''
    from aiq_swe_bench.predictors import register as register_predictors
    from aiq_swe_bench.predictors.predict_abc import SweBenchPredictorBase
    from aiq_swe_bench.predictors.predictor_registry import PredictorRegistry

    from aiq.builder.function_info import FunctionInfo
    from aiq.data_models.swe_bench_model import SWEBenchInput
    from aiq.data_models.swe_bench_model import SWEBenchOutput

    def _convert_input(input_str: str) -> SWEBenchInput:
        '''Convert a JSON string into an SWEBenchInput object.'''
        try:
            return SWEBenchInput.parse_raw(input_str)
        except Exception as e:
            raise ValueError(f"Invalid input format: {e}") from e

    def _convert_output(swe_bench_input: SWEBenchInput, model_patch: str) -> SWEBenchOutput:
        '''Convert model_patch to SWEBenchOutput object.'''
        return SWEBenchOutput(
            instance_id=swe_bench_input.instance_id,
            model_name_or_path="nv_predictor",
            model_patch=model_patch,
        )

    def _get_predictor() -> SweBenchPredictorBase:
        '''Fetch the predictor based on the prediction type such as gold, full etc.'''
        return PredictorRegistry.get(config.predictor.static_type())

    async def _response_fn(swe_bench_input_str: str) -> SWEBenchOutput:
        '''Response function called for each SWE Bench instance'''
        swe_bench_input = _convert_input(swe_bench_input_str)
        # Call the predict function
        model_patch = await _workflow.predict_fn(swe_bench_input)
        return _convert_output(swe_bench_input, model_patch)

    _predictor_callable = _get_predictor()
    _workflow = _predictor_callable(config, builder)

    yield FunctionInfo.create(single_fn=_response_fn)
