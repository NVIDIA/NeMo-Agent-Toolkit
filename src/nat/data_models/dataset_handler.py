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

import importlib
import json
import typing
from collections.abc import Callable
from pathlib import Path

import pandas as pd
from pydantic import BaseModel
from pydantic import Discriminator
from pydantic import FilePath
from pydantic import Tag

from nat.data_models.common import BaseModelRegistryTag
from nat.data_models.common import SerializableSecretStr
from nat.data_models.common import TypedBaseModel


class EvalS3Config(BaseModel):

    endpoint_url: str | None = None
    region_name: str | None = None
    bucket: str
    access_key: SerializableSecretStr
    secret_key: SerializableSecretStr


class EvalFilterEntryConfig(BaseModel):
    # values are lists of allowed/blocked values
    field: dict[str, list[str | int | float]] = {}


class EvalFilterConfig(BaseModel):
    allowlist: EvalFilterEntryConfig | None = None
    denylist: EvalFilterEntryConfig | None = None


class EvalDatasetStructureConfig(BaseModel):
    disable: bool = False
    question_key: str = "question"
    answer_key: str = "answer"
    generated_answer_key: str = "generated_answer"
    trajectory_key: str = "intermediate_steps"
    expected_trajectory_key: str = "expected_intermediate_steps"


# Base model
class EvalDatasetBaseConfig(TypedBaseModel, BaseModelRegistryTag):

    id_key: str = "id"
    structure: EvalDatasetStructureConfig = EvalDatasetStructureConfig()

    # Filters
    filter: EvalFilterConfig | None = EvalFilterConfig()

    s3: EvalS3Config | None = None

    remote_file_path: str | None = None  # only for s3
    file_path: Path | str = Path(".tmp/nat/examples/default/default.json")


class EvalDatasetJsonConfig(EvalDatasetBaseConfig, name="json"):

    @staticmethod
    def parser() -> tuple[Callable, dict]:
        return pd.read_json, {}


def read_jsonl(file_path: FilePath):
    with open(file_path, encoding='utf-8') as f:
        data = [json.loads(line) for line in f]
    return pd.DataFrame(data)


class EvalDatasetJsonlConfig(EvalDatasetBaseConfig, name="jsonl"):

    @staticmethod
    def parser() -> tuple[Callable, dict]:
        return read_jsonl, {}


class EvalDatasetCsvConfig(EvalDatasetBaseConfig, name="csv"):

    @staticmethod
    def parser() -> tuple[Callable, dict]:
        return pd.read_csv, {}


class EvalDatasetParquetConfig(EvalDatasetBaseConfig, name="parquet"):

    @staticmethod
    def parser() -> tuple[Callable, dict]:
        return pd.read_parquet, {}


class EvalDatasetXlsConfig(EvalDatasetBaseConfig, name="xls"):

    @staticmethod
    def parser() -> tuple[Callable, dict]:
        return pd.read_excel, {"engine": "openpyxl"}


class EvalDatasetCustomConfig(EvalDatasetBaseConfig, name="custom"):
    """
    Configuration for custom dataset type that allows users to specify
    a custom Python function to transform their dataset into EvalInput format.
    """

    function: str  # Direct import path to function, format: "module.path.function_name"
    kwargs: dict[str, typing.Any] = {}  # Additional arguments to pass to the custom function

    def parser(self) -> tuple[Callable, dict]:
        """
        Load and return the custom function for dataset transformation.

        Returns:
            Tuple of (custom_function, kwargs) where custom_function transforms
            a dataset file into an EvalInput object.
        """
        custom_function = self._load_custom_function()
        return custom_function, self.kwargs

    def _load_custom_function(self) -> Callable:
        """
        Import and return the custom function using standard Python import path.
        """
        if not self.function:
            raise ValueError("Function path cannot be empty")

        # Split the function path to get module and function name
        module_path, function_name = self.function.rsplit(".", 1)

        # Import the module
        module = importlib.import_module(module_path)

        # Get the function from the module
        if not hasattr(module, function_name):
            raise AttributeError(f"Function '{function_name}' not found in module '{module_path}'")

        custom_function = getattr(module, function_name)

        if not callable(custom_function):
            raise ValueError(f"'{self.function}' is not callable")

        return custom_function


# Union model with discriminator
EvalDatasetConfig = typing.Annotated[
    typing.Annotated[EvalDatasetJsonConfig, Tag(EvalDatasetJsonConfig.static_type())]
    | typing.Annotated[EvalDatasetCsvConfig, Tag(EvalDatasetCsvConfig.static_type())]
    | typing.Annotated[EvalDatasetXlsConfig, Tag(EvalDatasetXlsConfig.static_type())]
    | typing.Annotated[EvalDatasetParquetConfig, Tag(EvalDatasetParquetConfig.static_type())]
    | typing.Annotated[EvalDatasetJsonlConfig, Tag(EvalDatasetJsonlConfig.static_type())]
    | typing.Annotated[EvalDatasetCustomConfig, Tag(EvalDatasetCustomConfig.static_type())],
    Discriminator(TypedBaseModel.discriminator)]
