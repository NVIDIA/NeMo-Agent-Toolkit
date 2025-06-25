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


class EvaluationRunConfig(BaseModel):
    """
    Parameters used for a single evaluation run.
    """
    config_file: Path
    dataset: str | None  # dataset file path can be specified in the config file
    result_json_path: str = "$"
    skip_workflow: bool = False
    skip_completed_entries: bool = False
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


class MultiEvalutionRunConfig(BaseModel):
    """
    Parameters used for a multi-evaluation run.
    This includes a base config and a dict of overrides. The key is an id of
    any type.
    Each pass loads the base config and runs to completion before the next pass
    starts.
    """
    base_config: EvaluationRunConfig
    overrides: dict[typing.Any, str]


class MultiEvaluationRunOutput(BaseModel):
    """
    Output of a multi-evaluation run.
    The results per-pass are accumulated in the evaluation_runs dict.
    """
    evaluation_runs: dict[typing.Any, EvaluationRunOutput]
