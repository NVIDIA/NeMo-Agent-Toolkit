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

from pathlib import Path

from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field
from pydantic import model_validator

from nat.data_models.common import SerializableSecretStr


class EvalS3Config(BaseModel):

    model_config = ConfigDict(extra="forbid")

    endpoint_url: str | None = None
    region_name: str | None = None
    bucket: str
    access_key: SerializableSecretStr
    secret_key: SerializableSecretStr


class EvalFilterEntryConfig(BaseModel):

    model_config = ConfigDict(extra="forbid")

    # values are lists of allowed/blocked values
    field: dict[str, list[str | int | float]] = {}


class EvalFilterConfig(BaseModel):

    model_config = ConfigDict(extra="forbid")

    allowlist: EvalFilterEntryConfig | None = None
    denylist: EvalFilterEntryConfig | None = None


class EvalDatasetStructureConfig(BaseModel):

    model_config = ConfigDict(extra="forbid")

    disable: bool = False
    question_key: str = "question"
    answer_key: str = "answer"
    generated_answer_key: str = "generated_answer"
    trajectory_key: str = "intermediate_steps"
    expected_trajectory_key: str = "expected_intermediate_steps"


class EvalDatasetConfig(BaseModel):
    """Configuration for evaluation datasets.

    Supports two modes:
    - file_path: Read from a local file. Internally uses a transient FileObjectStore.
    - object_store + key: Read from a named ObjectStore (S3, Redis, etc.).

    Format is inferred from extension when not specified.
    """

    model_config = ConfigDict(extra="forbid")

    # Data source (provide one)
    file_path: Path | str | None = Field(default=None, description="Path to a local dataset file.")
    object_store: str | None = Field(default=None, description="Name of a configured ObjectStore.")
    key: str | None = Field(default=None, description="Key within the ObjectStore.")

    # Format
    format: str | None = Field(default=None,
                               description="Data format: csv, json, jsonl, parquet, xls. Inferred if omitted.")

    # Legacy fields (accepted for backwards compatibility)
    function: str | None = Field(default=None, description="Custom parser function (legacy).")
    kwargs: dict | None = Field(default=None, description="Custom parser kwargs (legacy).")

    # Schema
    id_key: str = Field(default="id", description="Column name for row IDs.")
    structure: EvalDatasetStructureConfig = Field(default_factory=EvalDatasetStructureConfig)
    filter: EvalFilterConfig | None = Field(default_factory=EvalFilterConfig)

    @model_validator(mode="before")
    @classmethod
    def handle_legacy_type(cls, values):
        """Accept legacy _type field for backwards compatibility.

        Old configs used ``_type: json`` (or csv, etc.) to select a DatasetLoader.
        We convert this to the ``format`` field so existing configs keep working.
        """
        if isinstance(values, dict) and "_type" in values:
            legacy_type = values.pop("_type")
            # Only set format if not already explicitly provided
            if "format" not in values or values["format"] is None:
                values["format"] = legacy_type
        return values

    @model_validator(mode="after")
    def validate_source(self):
        if self.file_path is None and self.object_store is None:
            raise ValueError("Must specify either 'file_path' or 'object_store'.")
        if self.object_store is not None and self.key is None:
            raise ValueError("Must specify 'key' when using 'object_store'.")
        return self
