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

import typing
from pathlib import Path

from pydantic import BaseModel

from aiq.eval.evaluator.evaluator_model import EvalInput
from aiq.eval.evaluator.evaluator_model import EvalOutput
from aiq.eval.usage_stats import UsageStats
from aiq.profiler.data_models import ProfilerResults


class EvaluationRunConfig(BaseModel):
    """
    Parameters used for a single evaluation run.
    """
    config_file: Path
    dataset: str | None = None  # dataset file path can be specified in the config file
    result_json_path: str = "$"
    skip_workflow: bool = False
    skip_completed_entries: bool = True
    endpoint: str | None = None  # only used when running the workflow remotely
    endpoint_timeout: int = 300
    reps: int = 1
    override: tuple[tuple[str, str], ...] = ()
    write_output: bool = True  # If false, the output will not be written to the output directory


class EvaluationRunOutput(BaseModel):
    """
    Output of a single evaluation run.
    """
    workflow_output_file: Path | None
    evaluator_output_files: list[Path]
    workflow_interrupted: bool

    eval_input: EvalInput
    evaluation_results: list[tuple[str, EvalOutput]]
    usage_stats: UsageStats | None = None
    profiler_results: ProfilerResults


class MultiEvaluationRunConfig(BaseModel):
    """
    Parameters used for a multi-evaluation run.
    This includes a base config and a dict of overrides. The key is an id of
    any type.
    Each pass loads the base config and runs to completion before the next pass
    starts.
    """
    base_config: EvaluationRunConfig
    overrides: dict[typing.Any, tuple[tuple[str, str], ...]]
    # todo: make this independent of the parameter
    reps_per_run: dict[typing.Any, int]
    write_output: bool = True


class MultiEvaluationRunOutput(BaseModel):
    """
    Output of a multi-evaluation run.
    The results per-pass are accumulated in the evaluation_runs dict.
    """
    evaluation_runs: dict[typing.Any, EvaluationRunOutput]


class CalcRunnerConfig(BaseModel):
    """
    Parameters used for a calc runner.
    """
    config_file: Path
    reps: int = 1
    concurrencies: list[int]

    target_p95_latency: float
    target_p95_workflow_runtime: float
    target_users: int

    test_gpu_count: int
    test_gpu_type: str | None = None

    plot_output_dir: Path | None = None


class MetricPerConcurrency(BaseModel):
    """
    Metrics per concurrency.
    """
    p95_latency: float
    p95_workflow_runtime: float
    total_runtime: float


class GPUEstimation(BaseModel):
    """
    GPU estimation.
    """
    min_required_gpus: float = -1
    p95_required_gpus: float = -1
    # gpu estimates per concurrency
    gpu_estimates: dict[int, float] = {}


class CalcRunnerOutput(BaseModel):
    """
    Output of the calc runner.
    """
    gpu_estimation: GPUEstimation
    # metric per tested concurrency
    metrics_per_concurrency: dict[int, MetricPerConcurrency]
