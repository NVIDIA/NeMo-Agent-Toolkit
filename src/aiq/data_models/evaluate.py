# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from pydantic import Discriminator

from aiq.data_models.common import TypedBaseModel
from aiq.data_models.dataset_handler import EvalDatasetConfig
from aiq.data_models.evaluator import EvaluatorBaseConfig
from aiq.data_models.profiler import ProfilerConfig


class EvalGeneralConfig(BaseModel):
    max_concurrency: int = 8

    # Output directory for the workflow and evaluation results
    output_dir: Path = Path("/tmp/aiq/examples/default/")

    # Dataset for running the workflow and evaluating
    dataset: EvalDatasetConfig | None = None

    # Inference profiler
    profiler: ProfilerConfig | None = None


class EvalConfig(BaseModel):

    # General Evaluation Options
    general: EvalGeneralConfig = EvalGeneralConfig()

    # Evaluators
    evaluators: dict[str, EvaluatorBaseConfig] = {}

    @classmethod
    def rebuild_annotations(cls):

        from aiq.cli.type_registry import GlobalTypeRegistry  # pylint: disable=cyclic-import

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
