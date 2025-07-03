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
    # number of passes at each concurrency, if 0 the dataset is adjusted to a multiple of the
    # concurrency. The is only used if adjust_dataset_size is true
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
    config_file: Path | None = None
    # if true workflow is not run, instead results from previous runs are used to estimate the
    # GPU count
    offline_mode: bool = False

    # number of passes at each concurrency, if 0 the dataset is adjusted to a multiple of the
    # concurrency
    num_passes: int = 0
    # concurrency values to test
    concurrencies: list[int] = [1, 2, 4, 8]

    # Targets for GPU estimation
    target_llm_latency_p95: float = 0
    target_workflow_runtime_p95: float = 0
    target_users: int = 0

    # Test setup information needed for GPU estimation
    test_gpu_count: int = 0

    # output directory for results
    output_dir: Path | None = None
    # if true, the job is stored in a new subdirectory of the output directory
    append_job: bool = False


class SizingMetricPerItem(BaseModel):
    """
    Metrics per item.
    """
    # LLM latency
    llm_latency: float
    # workflow runtime
    workflow_runtime: float


class SizingMetricsPerConcurrency(BaseModel):
    """
    Metrics per concurrency.
    """
    # p95 LLM latency
    llm_latency_p95: float
    # p95 workflow runtime
    workflow_runtime_p95: float
    # total workflow runtime
    total_runtime: float
    # per item metrics, key is the dataset entry id
    per_item_metrics: dict[typing.Any, SizingMetricPerItem]


class GPUEstimatesPerConcurrency(BaseModel):
    """
    GPU estimates per concurrency.
    """
    # gpu estimates per concurrency based on the workflow runtime
    gpu_estimate_by_wf_runtime: float | None = None
    # gpu estimates per concurrency based on the LLM latency
    gpu_estimate_by_llm_latency: float | None = None
    # gpu estimates per concurrency based on the number of users
    gpu_estimate: float | None = None


class GPUEstimates(BaseModel):
    """
    GPU estimates.
    """
    # minimum number of GPUs required based on the highest concurrency that passed the SLA
    gpu_estimate_min: float | None = None
    # 95th percentile of the number of GPUs required based on the highest concurrency that passed the SLA
    gpu_estimate_p95: float | None = None


class OutOfRangeRunsPerConcurrency(BaseModel):
    """
    Out of range runs.
    """
    # number of failed runs
    number_failed_runs: int = 0
    # number of runs that are greater than the target latency
    num_runs_greater_than_target_latency: int = 0
    # number of runs that are greater than the target runtime
    num_runs_greater_than_target_runtime: int = 0


class CalcRunnerOutput(BaseModel):
    """
    Output of the calc runner.
    """
    # GPU estimates, calculated online or offline
    gpu_estimates: GPUEstimates = GPUEstimates()

    # GPU estimates by concurrency, calculated online or offline
    gpu_estimates_per_concurrency: dict[int, GPUEstimatesPerConcurrency] = {}

    # Out of range runs, gathered based on the targets per concurrency. Calculated online or offline
    out_of_range_runs_per_concurrency: dict[int, OutOfRangeRunsPerConcurrency] = {}

    # Sizing metrics per tested concurrency. This information can only be gathered online.
    # It can be used offline for post-processing and GPU estimation.
    sizing_metrics_per_concurrency: dict[int, SizingMetricsPerConcurrency] = {}
