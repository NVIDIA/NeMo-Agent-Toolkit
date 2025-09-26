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

from pathlib import Path

from pydantic import BaseModel

from nat.eval.evaluator.evaluator_model import EvalInput
from nat.eval.evaluator.evaluator_model import EvalOutput
from nat.eval.usage_stats import UsageStats
from nat.profiler.data_models import ProfilerResults


class EvaluationRunConfig(BaseModel):
    """
    Parameters used for a single evaluation run.
    """
    config_file: Path | BaseModel
    dataset: str | None = None  # dataset file path can be specified in the config file
    result_json_path: str = "$"
    skip_workflow: bool = False
    skip_completed_entries: bool = False
    endpoint: str | None = None  # only used when running the workflow remotely
    endpoint_timeout: int = 300
    reps: int = 1
    override: tuple[tuple[str, str], ...] = ()
    # If false, the output will not be written to the output directory. This is
    # useful when running evaluation via another tool.
    write_output: bool = True
    # if true, the dataset is adjusted to a multiple of the concurrency
    adjust_dataset_size: bool = False
    # number of passes at each concurrency, if 0 the dataset is adjusted to a multiple of the
    # concurrency. The is only used if adjust_dataset_size is true
    num_passes: int = 0
    # timeout for waiting for trace export tasks to complete
    export_timeout: float = 60.0


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
