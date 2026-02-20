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

import pandas as pd

from nat.builder.builder import EvalBuilder
from nat.builder.dataset_loader import DatasetLoaderInfo
from nat.cli.register_workflow import register_dataset_loader
from nat.data_models.dataset_handler import EvalDatasetCsvConfig
from nat.data_models.dataset_handler import EvalDatasetCustomConfig
from nat.data_models.dataset_handler import EvalDatasetJsonConfig
from nat.data_models.dataset_handler import EvalDatasetJsonlConfig
from nat.data_models.dataset_handler import EvalDatasetParquetConfig
from nat.data_models.dataset_handler import EvalDatasetXlsConfig
from nat.data_models.dataset_handler import read_jsonl


@register_dataset_loader(config_type=EvalDatasetJsonConfig)
async def register_json_dataset_loader(config: EvalDatasetJsonConfig, builder: EvalBuilder):
    yield DatasetLoaderInfo(config=config, load_fn=pd.read_json, description="JSON file dataset loader")


@register_dataset_loader(config_type=EvalDatasetJsonlConfig)
async def register_jsonl_dataset_loader(config: EvalDatasetJsonlConfig, builder: EvalBuilder):
    yield DatasetLoaderInfo(config=config, load_fn=read_jsonl, description="JSONL file dataset loader")


@register_dataset_loader(config_type=EvalDatasetCsvConfig)
async def register_csv_dataset_loader(config: EvalDatasetCsvConfig, builder: EvalBuilder):
    yield DatasetLoaderInfo(config=config, load_fn=pd.read_csv, description="CSV file dataset loader")


@register_dataset_loader(config_type=EvalDatasetParquetConfig)
async def register_parquet_dataset_loader(config: EvalDatasetParquetConfig, builder: EvalBuilder):
    yield DatasetLoaderInfo(config=config, load_fn=pd.read_parquet, description="Parquet file dataset loader")


@register_dataset_loader(config_type=EvalDatasetXlsConfig)
async def register_xls_dataset_loader(config: EvalDatasetXlsConfig, builder: EvalBuilder):

    def load_excel(file_path, **kwargs):
        return pd.read_excel(file_path, engine="openpyxl", **kwargs)

    yield DatasetLoaderInfo(config=config, load_fn=load_excel, description="Excel file dataset loader")


@register_dataset_loader(config_type=EvalDatasetCustomConfig)
async def register_custom_dataset_loader(config: EvalDatasetCustomConfig, builder: EvalBuilder):
    custom_fn, kwargs = config.parser()

    def load_custom(file_path, **extra_kwargs):
        merged = {**kwargs, **extra_kwargs}
        return custom_fn(file_path=file_path, **merged)

    yield DatasetLoaderInfo(config=config, load_fn=load_custom, description="Custom function dataset loader")
