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

from nat.data_models.atif import ATIFAgentConfig
from nat.data_models.atif import ATIFTrajectory
from nat.data_models.evaluator import EvalOutputItem
from nat.plugins.eval.evaluator.atif_base_evaluator import AtifBaseEvaluator
from nat.plugins.eval.evaluator.atif_evaluator import AtifEvalSample


def _sample(item_id: str, expected: str, generated: str) -> AtifEvalSample:
    trajectory = ATIFTrajectory(session_id=f"session-{item_id}",
                                agent=ATIFAgentConfig(name="test-agent", version="0.0.0"))
    return AtifEvalSample(item_id=item_id, trajectory=trajectory, expected_output_obj=expected, output_obj=generated)


class _LengthRatioAtifEvaluator(AtifBaseEvaluator):

    async def evaluate_atif_item(self, sample: AtifEvalSample) -> EvalOutputItem:
        expected = str(sample.expected_output_obj or "")
        generated = str(sample.output_obj or "")
        score = round(len(generated) / max(len(expected), 1), 2)
        return EvalOutputItem(id=sample.item_id, score=score, reasoning={"score": score})


class _ConcurrencyProbeAtifEvaluator(AtifBaseEvaluator):

    def __init__(self, max_concurrency: int):
        super().__init__(max_concurrency=max_concurrency)
        self._active = 0
        self.peak_active = 0

    async def evaluate_atif_item(self, sample: AtifEvalSample) -> EvalOutputItem:
        self._active += 1
        self.peak_active = max(self.peak_active, self._active)
        try:
            await asyncio.sleep(0.01)
            return EvalOutputItem(id=sample.item_id, score=1.0, reasoning={"score": 1.0})
        finally:
            self._active -= 1


async def test_atif_base_evaluator_computes_average_score():
    evaluator = _LengthRatioAtifEvaluator(max_concurrency=2)
    samples = [
        _sample("1", "abcd", "abcd"),
        _sample("2", "abcd", "ab"),
    ]
    output = await evaluator.evaluate_atif_fn(samples)

    assert len(output.eval_output_items) == 2
    assert output.average_score == 0.75


async def test_atif_base_evaluator_uses_bounded_concurrency():
    evaluator = _ConcurrencyProbeAtifEvaluator(max_concurrency=2)
    samples = [_sample(str(i), "x", "x") for i in range(6)]

    await evaluator.evaluate_atif_fn(samples)

    assert evaluator.peak_active <= 2
    assert evaluator.peak_active > 1


async def test_atif_base_evaluator_processes_all_samples_when_remainder_exists():
    """Ensure semaphore-based batching does not drop tail samples."""
    evaluator = _ConcurrencyProbeAtifEvaluator(max_concurrency=4)
    samples = [_sample(str(i), "x", "x") for i in range(10)]  # 10 % 4 != 0

    output = await evaluator.evaluate_atif_fn(samples)

    assert len(output.eval_output_items) == len(samples)
    returned_ids = {str(item.id) for item in output.eval_output_items}
    expected_ids = {str(sample.item_id) for sample in samples}
    assert returned_ids == expected_ids
    assert evaluator.peak_active <= 4
