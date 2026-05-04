# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from __future__ import annotations

from typing import Self

from pydantic import ConfigDict
from pydantic import model_validator

from nat.builder.builder import EvalBuilder
from nat.builder.dataset_loader import DatasetLoaderInfo
from nat.cli.register_workflow import register_dataset_loader
from nat.data_models.dataset_handler import EvalDatasetBaseConfig


class EvalDatasetLangSmithConfig(EvalDatasetBaseConfig, name="langsmith"):
    """Load evaluation dataset from LangSmith by dataset ID or name."""

    model_config = ConfigDict(extra="forbid")

    dataset_id: str | None = None
    dataset_name: str | None = None
    input_key: str = "input"
    output_key: str = "output"
    split: str | None = None
    as_of: str | None = None
    limit: int | None = None

    @model_validator(mode="after")
    def _require_id_or_name(self) -> Self:
        if not self.dataset_id and not self.dataset_name:
            raise ValueError("At least one of 'dataset_id' or 'dataset_name' must be provided")
        return self

    def parser(self) -> tuple:
        from .langsmith import load_langsmith_dataset

        return load_langsmith_dataset, {
            "dataset_id": self.dataset_id,
            "dataset_name": self.dataset_name,
            "input_key": self.input_key,
            "output_key": self.output_key,
            "question_col": self.structure.question_key,
            "answer_col": self.structure.answer_key,
            "id_col": self.id_key,
            "split": self.split,
            "as_of": self.as_of,
            "limit": self.limit,
        }


@register_dataset_loader(config_type=EvalDatasetLangSmithConfig)
async def register_langsmith_dataset_loader(config: EvalDatasetLangSmithConfig, builder: EvalBuilder):
    from .langsmith import load_langsmith_dataset

    _, kwargs = config.parser()

    def load_fn(file_path, **extra_kwargs):
        merged = {**kwargs, **extra_kwargs}
        return load_langsmith_dataset(file_path, **merged)

    yield DatasetLoaderInfo(config=config, load_fn=load_fn, description="LangSmith dataset loader")
