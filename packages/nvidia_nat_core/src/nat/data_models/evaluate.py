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

import typing
from enum import Enum
from pathlib import Path

from pydantic import BaseModel
from pydantic import Discriminator
from pydantic import Field
from pydantic import model_validator

from nat.data_models.common import TypedBaseModel
from nat.data_models.dataset_handler import EvalDatasetConfig
from nat.data_models.dataset_handler import EvalS3Config
from nat.data_models.evaluator import EvaluatorBaseConfig
from nat.data_models.intermediate_step import IntermediateStepType
from nat.data_models.profiler import ProfilerConfig


class JobEvictionPolicy(str, Enum):
    """Policy for evicting old jobs when max_jobs is exceeded."""
    TIME_CREATED = "time_created"
    TIME_MODIFIED = "time_modified"


class EvalCustomScriptConfig(BaseModel):
    script: Path = Field(description="Path to the script to run.")

    kwargs: dict[str, str] = Field(default_factory=dict, description="Keyword arguments to pass to the script.")


class JobManagementConfig(BaseModel):
    append_job_id_to_output_dir: bool = Field(
        default=False, description="Whether to append a unique job ID to the output directory for each run.")

    max_jobs: int = Field(default=0,
                          description="Maximum number of jobs to keep in the output directory. "
                          "Oldest jobs will be evicted. A value of 0 means no limit.")

    eviction_policy: JobEvictionPolicy = Field(default=JobEvictionPolicy.TIME_CREATED,
                                               description="Policy for evicting old jobs.")


class EvalOutputConfig(BaseModel):
    dir: Path = Field(default=Path("./.tmp/nat/examples/default/"),
                      description="Output directory for the workflow and evaluation results.")

    remote_dir: str | None = Field(default=None, description="S3 prefix for the workflow and evaluation results.")

    custom_pre_eval_process_function: str | None = Field(
        default=None,
        description="Custom function to pre-evaluation process the eval input. Format: 'module.path.function_name'.")

    custom_scripts: dict[str, EvalCustomScriptConfig] = Field(
        default_factory=dict, description="Custom scripts to run after the workflow and evaluation results are saved.")

    s3: EvalS3Config | None = Field(default=None,
                                    description="S3 config for uploading the contents of the output directory.")

    cleanup: bool = Field(default=True,
                          description="Whether to cleanup the output directory before running the workflow.")

    job_management: JobManagementConfig = Field(default_factory=JobManagementConfig,
                                                description="Job management configuration (job id, eviction, etc.).")

    workflow_output_step_filter: list[IntermediateStepType] | None = Field(
        default=None, description="Filter for the workflow output steps.")


class EvalGeneralConfig(BaseModel):
    max_concurrency: int = Field(default=8, description="Maximum number of concurrent workflow executions.")

    workflow_alias: str | None = Field(
        default=None,
        description="Workflow alias for displaying in evaluation UI. If not provided, the workflow type will be used.")

    output_dir: Path = Field(default=Path("./.tmp/nat/examples/default/"),
                             description="Output directory for the workflow and evaluation results.")

    output: EvalOutputConfig | None = Field(default=None,
                                            description="Output configuration. If present, overrides output_dir.")

    dataset: EvalDatasetConfig | None = Field(
        default=None, description="Dataset configuration for running the workflow and evaluating.")

    profiler: ProfilerConfig | None = Field(default=None, description="Inference profiler configuration.")

    validate_llm_endpoints: bool = Field(
        default=False,
        description="When enabled, validates that all LLM endpoints are accessible before starting evaluation. "
        "This catches deployment issues early (e.g., 404 errors from canceled training jobs). "
        "Recommended for production workflows.")

    per_input_user_id: bool = Field(
        default=True,
        description="When enabled, generates a unique user_id for each eval item. For per-user workflows, "
        "this creates a fresh workflow instance per eval item, resetting all stateful tools to their "
        "initial state. Set to False to disable this behavior.")

    # overwrite the output_dir with the output config if present
    @model_validator(mode="before")
    @classmethod
    def override_output_dir(cls, values):
        if values.get("output") and values["output"].get("dir"):
            values["output_dir"] = values["output"]["dir"]
        return values


class EvalConfig(BaseModel):
    general: EvalGeneralConfig = Field(default_factory=EvalGeneralConfig, description="General evaluation options.")

    evaluators: dict[str, EvaluatorBaseConfig] = Field(default_factory=dict, description="Evaluators configuration.")

    @classmethod
    def rebuild_annotations(cls):

        from nat.cli.type_registry import GlobalTypeRegistry

        type_registry = GlobalTypeRegistry.get()

        EvaluatorsAnnotation = dict[str,
                                    typing.Annotated[type_registry.compute_annotation(EvaluatorBaseConfig),
                                                     Discriminator(TypedBaseModel.discriminator)]]

        should_rebuild = False

        evaluators_field = cls.model_fields.get("evaluators")
        if evaluators_field is not None and evaluators_field.annotation != EvaluatorsAnnotation:
            evaluators_field.annotation = EvaluatorsAnnotation
            should_rebuild = True

        if (should_rebuild):
            cls.model_rebuild(force=True)
