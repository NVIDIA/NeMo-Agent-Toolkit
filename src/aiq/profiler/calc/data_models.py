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


class LinearFitResult(BaseModel):
    """Result of linear regression including slope, intercept, and quality metrics."""
    slope: float
    intercept: float
    r_squared: float
    outliers_removed: int


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


class GPUEstimates(BaseModel):
    """
    GPU estimates. These use the slope of the time vs concurrency to
    estimate the number of GPUs required.
    """
    # GPU estimate based on the slope of the runtime vs concurrency
    gpu_estimate_by_wf_runtime: float | None = None
    # GPU estimate based on the slope of the latency vs concurrency
    gpu_estimate_by_llm_latency: float | None = None


class GPUEstimatesPerConcurrency(BaseModel):
    """
    GPU estimates per concurrency. These use a multiplier based on the
    target users and the test concurrency to estimate the number of GPUs required.
    """
    # gpu estimates per concurrency based on the workflow runtime
    gpu_estimate_by_wf_runtime: float | None = None
    # gpu estimates per concurrency based on the LLM latency
    gpu_estimate_by_llm_latency: float | None = None


class OutOfRangeRunsPerConcurrency(BaseModel):
    """
    Out of range runs.
    """
    # number of failed runs, no output or intermediate steps are available for these runs
    number_failed_runs: int = 0
    # number of runs that are greater than the target latency
    num_runs_greater_than_target_latency: int = 0
    # number of runs that are greater than the target runtime
    num_runs_greater_than_target_runtime: int = 0


class CalcRunnerOutput(BaseModel):
    """
    Output of the calc runner.
    """
    # GPU estimates based on the slope of the time vs concurrency, calculated online or offline
    gpu_estimates: GPUEstimates = GPUEstimates()

    # GPU estimates by concurrency, calculated online or offline
    gpu_estimates_per_concurrency: dict[int, GPUEstimatesPerConcurrency] = {}

    # Out of range runs, gathered based on the targets per concurrency. Calculated online or offline
    out_of_range_runs_per_concurrency: dict[int, OutOfRangeRunsPerConcurrency] = {}

    # Sizing metrics per tested concurrency. This information can only be gathered online.
    # It can be used offline for post-processing and GPU estimation.
    sizing_metrics_per_concurrency: dict[int, SizingMetricsPerConcurrency] = {}
