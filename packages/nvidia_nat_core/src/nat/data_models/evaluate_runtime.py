# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Runtime-only evaluation models used by `nat eval` programmatic execution."""

from pathlib import Path

from pydantic import BaseModel
from pydantic import Field

from nat.data_models.evaluator import EvalInput
from nat.data_models.evaluator import EvalOutput


class EndpointRetryConfig(BaseModel):
    """Configuration for HTTP retry behavior on remote workflow endpoints."""

    do_auto_retry: bool = Field(
        default=True,
        description="Enable automatic retry on transient HTTP errors.",
    )
    max_retries: int = Field(
        default=3,
        ge=1,
        description="Maximum retry attempts.",
    )
    retry_status_codes: list[int] = Field(
        default=[429, 500, 502, 503, 504],
        description="HTTP status codes that trigger automatic retry.",
    )


class EvaluationRunConfig(BaseModel):
    """
    Parameters used for a single evaluation run. This is used by the `nat eval` command. It
    can also be used for programmatic evaluation.
    """

    config_file: Path | BaseModel = Field(
        ...,
        description="Path to the evaluation config file or a config model instance.",
    )
    dataset: str | None = Field(
        default=None,
        description="Dataset file path. Can also be specified in the config file.",
    )
    result_json_path: str = Field(
        default="$",
        description="JSONPath expression to extract the result from workflow output.",
    )
    skip_workflow: bool = Field(
        default=False,
        description="If true, skip workflow execution and use existing outputs.",
    )
    skip_completed_entries: bool = Field(
        default=False,
        description="If true, skip dataset entries that already have outputs.",
    )
    endpoint: str | None = Field(
        default=None,
        description="Remote workflow endpoint URL. Only used for remote execution.",
    )
    endpoint_timeout: int = Field(
        default=300,
        description="Timeout in seconds for remote workflow requests.",
    )
    endpoint_retry: EndpointRetryConfig = Field(
        default_factory=EndpointRetryConfig,
        description="Retry configuration for remote endpoint requests.",
    )
    reps: int = Field(
        default=1,
        description="Number of repetitions for each dataset entry.",
    )
    override: tuple[tuple[str, str], ...] = Field(
        default=(),
        description="Config overrides as key-value tuples.",
    )
    write_output: bool = Field(
        default=True,
        description="If false, output will not be written to disk. Useful when running via another tool.",
    )
    adjust_dataset_size: bool = Field(
        default=False,
        description="If true, adjust dataset size to a multiple of concurrency.",
    )
    num_passes: int = Field(
        default=0,
        description="Number of passes at each concurrency level. Only used if adjust_dataset_size is true.",
    )
    export_timeout: float = Field(
        default=60.0,
        description="Timeout in seconds for trace export tasks to complete.",
    )
    user_id: str = Field(
        default="nat_eval_user_id",
        description="User ID for the workflow session.",
    )


class UsageStatsLLM(BaseModel):
    """Token usage counters aggregated for one LLM."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_tokens: int = 0
    reasoning_tokens: int = 0
    total_tokens: int = 0


class UsageStatsItem(BaseModel):
    """Usage metrics for one evaluated input item."""

    usage_stats_per_llm: dict[str, UsageStatsLLM]
    total_tokens: int | None = None
    runtime: float = 0.0
    min_timestamp: float = 0.0
    max_timestamp: float = 0.0
    llm_latency: float = 0.0


class UsageStats(BaseModel):
    """Aggregated usage metrics across an evaluation run."""

    # key is EvalInputItem.id or equivalent identifier
    min_timestamp: float = 0.0
    max_timestamp: float = 0.0
    total_runtime: float = 0.0
    usage_stats_items: dict[object, UsageStatsItem] = {}


class InferenceMetricsModel(BaseModel):
    """Confidence intervals and percentiles for a sampled profiler metric."""

    n: int = Field(default=0, description="Number of samples")
    mean: float = Field(default=0, description="Mean of the samples")
    ninetieth_interval: tuple[float, float] = Field(default=(0, 0), description="90% confidence interval")
    ninety_fifth_interval: tuple[float, float] = Field(default=(0, 0), description="95% confidence interval")
    ninety_ninth_interval: tuple[float, float] = Field(default=(0, 0), description="99% confidence interval")
    p90: float = Field(default=0, description="90th percentile of the samples")
    p95: float = Field(default=0, description="95th percentile of the samples")
    p99: float = Field(default=0, description="99th percentile of the samples")


class WorkflowRuntimeMetrics(BaseModel):
    """p90/p95/p99 workflow runtimes across evaluation examples."""

    p90: float
    p95: float
    p99: float


class ProfilerResults(BaseModel):
    """High-level profiler output attached to an evaluation run."""

    workflow_runtime_metrics: WorkflowRuntimeMetrics | None = None
    llm_latency_ci: InferenceMetricsModel | None = None


class EvaluationRunOutput(BaseModel):
    """Output of a single evaluation run."""

    workflow_output_file: Path | None = Field(
        ...,
        description="Path to the workflow output JSON file.",
    )
    evaluator_output_files: list[Path] = Field(
        ...,
        description="Paths to evaluator output JSON files.",
    )
    workflow_interrupted: bool = Field(
        ...,
        description="True if the workflow was interrupted before completing all items.",
    )
    eval_input: EvalInput = Field(
        ...,
        description="Evaluation input containing all dataset items and their outputs.",
    )
    evaluation_results: list[tuple[str, EvalOutput]] = Field(
        ...,
        description="List of evaluator results as (evaluator_name, output) tuples.",
    )
    usage_stats: UsageStats | None = Field(
        default=None,
        description="LLM usage statistics collected during evaluation.",
    )
    profiler_results: ProfilerResults = Field(
        ...,
        description="Profiling results from the evaluation run.",
    )
    config_original_file: Path | None = Field(
        default=None,
        description="Path to the original config file written to output directory.",
    )
    config_effective_file: Path | None = Field(
        default=None,
        description="Path to the effective config file with overrides applied.",
    )
    config_metadata_file: Path | None = Field(
        default=None,
        description="Path to the config metadata file.",
    )
