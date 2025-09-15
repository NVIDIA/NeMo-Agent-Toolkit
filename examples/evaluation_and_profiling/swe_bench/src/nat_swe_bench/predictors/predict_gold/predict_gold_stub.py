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

from nat.data_models.swe_bench_model import SWEBenchInput
from nat_swe_bench.predictors.predict_abc import SweBenchPredictorBase
from nat_swe_bench.predictors.predictor_registry import register_predictor

logger = logging.getLogger(__name__)


@register_predictor("gold")
class SweBenchPredictor(SweBenchPredictorBase):

    async def predict_fn(self, swebench_input: SWEBenchInput) -> str:
        '''Bypass problem solving and return the input patch as the output patch.'''
        if self.config.predictor.verbose:
            logger.info("Swe-bench instance: %s", swebench_input.instance_id)
            logger.info("  repo: %s version %d", swebench_input.repo, swebench_input.version)
            logger.info(" problem: %s", swebench_input.problem_statement)

        logger.info("Providing gold patch for input instance: %s", swebench_input.instance_id)
        return swebench_input.patch
