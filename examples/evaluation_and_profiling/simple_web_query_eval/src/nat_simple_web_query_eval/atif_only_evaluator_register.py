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
"""ATIF-only custom evaluator example for NVIDIA NeMo Agent Toolkit."""

import math
from collections import Counter

from pydantic import Field

from nat.builder.builder import EvalBuilder
from nat.builder.evaluator import EvaluatorInfo
from nat.cli.register_workflow import register_evaluator
from nat.data_models.evaluator import EvalOutputItem
from nat.data_models.evaluator import EvaluatorBaseConfig
from nat.plugins.eval.evaluator.atif_base_evaluator import AtifBaseEvaluator
from nat.plugins.eval.evaluator.atif_evaluator import AtifEvalSample


class AtifCosineSimilarityEvaluatorConfig(EvaluatorBaseConfig, name="atif_cosine_similarity"):
    """Configuration for an ATIF-only cosine-similarity evaluator."""

    normalize_case: bool = Field(
        default=True,
        description="Whether to compare generated and expected outputs case-insensitively.",
    )


class AtifCosineSimilarityEvaluator(AtifBaseEvaluator):
    """Minimal ATIF-only evaluator that scores output and expected text similarity.

    Note:
        `AtifEvaluator` is a protocol used for structural typing (duck typing).
        This class does not need to explicitly inherit from `AtifEvaluator`;
        implementing `evaluate_atif_fn` with the expected signature is sufficient.
    """

    def __init__(self, normalize_case: bool = True, max_concurrency: int = 4) -> None:
        super().__init__(max_concurrency=max_concurrency)
        self._normalize_case = normalize_case

    def _normalize(self, value: object) -> str:
        text = "" if value is None else str(value).strip()
        return text.casefold() if self._normalize_case else text

    def _count_tool_calls(self, sample) -> int:
        steps = getattr(sample.trajectory, "steps", None) or []
        return sum(len(getattr(step, "tool_calls", None) or []) for step in steps)

    def _cosine_similarity(self, text_a: str, text_b: str) -> float:
        tokens_a = text_a.split()
        tokens_b = text_b.split()
        if not tokens_a or not tokens_b:
            return 0.0

        counts_a = Counter(tokens_a)
        counts_b = Counter(tokens_b)
        shared_tokens = set(counts_a) & set(counts_b)
        numerator = sum(counts_a[token] * counts_b[token] for token in shared_tokens)
        norm_a = math.sqrt(sum(value * value for value in counts_a.values()))
        norm_b = math.sqrt(sum(value * value for value in counts_b.values()))
        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0
        return numerator / (norm_a * norm_b)

    async def evaluate_atif_item(self, sample: AtifEvalSample) -> EvalOutputItem:
        """Score one ATIF sample using token cosine similarity."""
        expected_text = self._normalize(sample.expected_output_obj)
        generated_text = self._normalize(sample.output_obj)
        similarity_score = round(self._cosine_similarity(expected_text, generated_text), 2)
        tool_call_count = self._count_tool_calls(sample)
        return EvalOutputItem(id=sample.item_id,
                              score=similarity_score,
                              reasoning={
                                  "comparison": "cosine-similarity",
                                  "expected": expected_text,
                                  "generated": generated_text,
                                  "trajectory_tool_call_count": tool_call_count,
                              })


@register_evaluator(config_type=AtifCosineSimilarityEvaluatorConfig)
async def register_atif_cosine_similarity_evaluator(config: AtifCosineSimilarityEvaluatorConfig, _builder: EvalBuilder):
    """Register the ATIF-only cosine-similarity evaluator."""
    evaluator = AtifCosineSimilarityEvaluator(normalize_case=config.normalize_case,
                                              max_concurrency=_builder.get_max_concurrency())
    evaluator_info = EvaluatorInfo(config=config, description="ATIF-only cosine similarity custom evaluator")
    evaluator_info.evaluate_atif_fn = evaluator.evaluate_atif_fn
    yield evaluator_info
