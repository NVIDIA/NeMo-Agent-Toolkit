# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Configuration types for BYOB (Bring Your Own Benchmark) integration."""

from collections.abc import Callable
from typing import Any

from pydantic import Field

from nat.data_models.dataset_handler import EvalDatasetBaseConfig
from nat.data_models.evaluator import EvaluatorBaseConfig


class BYOBDatasetConfig(EvalDatasetBaseConfig, name="byob"):
    """Dataset config for BYOB benchmarks.

    Loads a dataset from a BYOB benchmark definition. The benchmark module
    and name are used to import the benchmark and access its dataset.
    """

    benchmark_module: str = Field(
        description="Python module path or file path to the benchmark definition "
        "(e.g. 'my_benchmarks.qa' or '/path/to/benchmark.py')",
    )
    benchmark_name: str = Field(
        description="Normalized benchmark name as registered with @benchmark decorator",
    )
    limit: int | None = Field(
        default=None,
        description="Limit number of dataset samples (for testing)",
    )

    def parser(self) -> tuple[Callable, dict]:
        from .dataset import load_byob_dataset
        return load_byob_dataset, {
            "benchmark_module": self.benchmark_module,
            "benchmark_name": self.benchmark_name,
            "limit": self.limit,
        }


class BYOBEvaluatorConfig(EvaluatorBaseConfig, name="byob_evaluator"):
    """Evaluator config for BYOB benchmarks.

    Calls bench.scorer_fn(ScorerInput(...)) directly; model_call_fn=None
    (safe for all built-in scorers like exact_match, contains, f1_token).
    """

    benchmark_module: str = Field(
        description="Python module path or file path to the benchmark definition",
    )
    benchmark_name: str = Field(
        description="Normalized benchmark name",
    )
    score_field: str = Field(
        default="correct",
        description="Key in scorer output dict to use as the primary score "
        "(e.g. 'correct' for exact_match, 'f1' for f1_token)",
    )
