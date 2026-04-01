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

from unittest.mock import MagicMock

from nat.data_models.atif import ATIFAgentConfig
from nat.data_models.atif import ATIFTrajectory
from nat.plugins.eval.evaluator.atif_evaluator import AtifEvalSample
from nat_simple_web_query_eval.atif_only_evaluator_register import AtifCosineSimilarityEvaluator
from nat_simple_web_query_eval.atif_only_evaluator_register import AtifCosineSimilarityEvaluatorConfig
from nat_simple_web_query_eval.atif_only_evaluator_register import register_atif_cosine_similarity_evaluator


async def test_register_atif_cosine_similarity_evaluator_exposes_only_atif_lane():
    config = AtifCosineSimilarityEvaluatorConfig()
    builder = MagicMock()
    builder.get_max_concurrency.return_value = 2
    async with register_atif_cosine_similarity_evaluator(config, builder) as evaluator_info:
        assert evaluator_info.evaluate_fn is None
        assert callable(evaluator_info.evaluate_atif_fn)


async def test_atif_cosine_similarity_evaluator_scores_items():
    evaluator = AtifCosineSimilarityEvaluator(normalize_case=True)
    trajectory = ATIFTrajectory(session_id="sample", agent=ATIFAgentConfig(name="test-agent", version="0.0.0"))
    samples = [
        AtifEvalSample(item_id="a", trajectory=trajectory, expected_output_obj="Alpha", output_obj="alpha"),
        AtifEvalSample(item_id="b", trajectory=trajectory, expected_output_obj="beta", output_obj="gamma"),
    ]

    output = await evaluator.evaluate_atif_fn(samples)

    assert output.average_score == 0.5
    assert len(output.eval_output_items) == 2
    assert output.eval_output_items[0].score == 1.0
    assert output.eval_output_items[1].score == 0.0
