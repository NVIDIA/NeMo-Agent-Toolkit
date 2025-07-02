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

    # if true, the dataset is adjusted to a multiple of the concurrency
    adjust_dataset_size: bool = False
    # number of passes at each concurrency, if 0 the dataset is adjusted to a multiple of the concurrency
    # only used if adjust_dataset_size is true
    num_passes: int = 0


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
    # number of passes at each concurrency
    num_passes: int = 0
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
    # base config - not needed in offline mode
    config_file: Path
    # if true workflow is not run, instead results from previous runs are used to estimate the GPU count
    offline_mode: bool = False

    # number of passes at each concurrency, if 0 the dataset is adjusted to a multiple of the concurrency
    num_passes: int = 0
    # concurrency values to test
    concurrencies: list[int]

    # Targets for GPU estimation
    target_p95_latency: float
    target_p95_workflow_runtime: float
    target_users: int

    # Information on the test setup needed for GPU estimation
    test_gpu_count: int
    test_gpu_type: str | None = None

    # output directory for results
    output_dir: Path | None = None
    # if true, the job is stored in a new subdirectory of the output directory
    append_job: bool = False


class MetricPerConcurrency(BaseModel):
    """
    Metrics per concurrency.
    """
    # p95 LLM latency
    p95_latency: float
    # p95 workflow runtime
    p95_workflow_runtime: float
    # total workflow runtime
    total_runtime: float


class GPUEstimation(BaseModel):
    """
    GPU estimation.
    """
    # minimum number of GPUs required based on the highest concurrency that passed the SLA
    min_required_gpus: float = -1
    # 95th percentile of the number of GPUs required based on the highest concurrency that passed the SLA
    p95_required_gpus: float = -1
    # gpu estimates per concurrency based on the workflow runtime
    gpu_estimates_by_wf_runtime: dict[int, float] = {}
    # gpu estimates per concurrency based on the LLM latency
    gpu_estimates_by_llm_latency: dict[int, float] = {}
    # gpu estimates per concurrency based on the number of users
    gpu_estimates: dict[int, float] = {}


class CalcRunnerOutput(BaseModel):
    """
    Output of the calc runner.
    """
    # GPU estimation
    gpu_estimation: GPUEstimation
    # metric per tested concurrency
    metrics_per_concurrency: dict[int, MetricPerConcurrency]
