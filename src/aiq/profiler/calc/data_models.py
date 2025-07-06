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
    # base config and endpoints (if remote)- not needed in offline mode
    config_file: Path | None = None
    # endpoint to use for the workflow, if not provided the workflow is run locally
    endpoint: str | None = None
    # timeout for the workflow
    endpoint_timeout: int = 300

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
    # if true, the metrics are eligible for slope-based GPU estimation
    eligible_for_slope_based_estimation: bool = False

    # p95 LLM latency
    llm_latency_p95: float
    # p95 workflow runtime
    workflow_runtime_p95: float
    # total workflow runtime
    total_runtime: float
    # per item metrics, key is the dataset entry id
    per_item_metrics: dict[typing.Any, SizingMetricPerItem] = {}


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
    These are ROUGH estimates and are not used for the final GPU estimation.
    The final GPU estimation is slope-based.
    """
    # gpu estimates per concurrency based on the workflow runtime
    gpu_estimate_by_wf_runtime: float | None = None
    # gpu estimates per concurrency based on the LLM latency
    gpu_estimate_by_llm_latency: float | None = None


class OutOfRangeItemsPerConcurrency(BaseModel):
    """
    Out of range items per concurrency.
    """
    # number of items that are greater than the target latency
    num_items_greater_than_target_latency: int = 0
    # number of items that are greater than the target runtime
    num_items_greater_than_target_runtime: int = 0
    # if the workflow was interrupted that pass cannot be used
    workflow_interrupted: bool = False


class CalcRunnerOutputPerConcurrency(BaseModel):
    """
    Output of the calc runner per concurrency.
    """
    # ROUGH GPU estimates per concurrency: these are not used for the final GPU estimation
    # they are only available for information purposes
    gpu_estimates: GPUEstimatesPerConcurrency = GPUEstimatesPerConcurrency()
    # Out of range runs per concurrency
    out_of_range_runs: OutOfRangeItemsPerConcurrency = OutOfRangeItemsPerConcurrency()
    # Sizing metrics per concurrency
    sizing_metrics: SizingMetricsPerConcurrency = SizingMetricsPerConcurrency()


class CalcRunnerOutput(BaseModel):
    """
    Output of the calc runner.
    """
    # GPU estimates based on the slope of the time vs concurrency, calculated online or offline
    gpu_estimates: GPUEstimates = GPUEstimates()

    # Per-concurrency data (GPU estimates, out-of-range runs, and sizing metrics)
    per_concurrency_data: dict[int, CalcRunnerOutputPerConcurrency] = {}
