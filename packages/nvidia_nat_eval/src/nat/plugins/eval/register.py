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

# flake8: noqa

# Dataset loaders
from .dataset_loader.register import register_csv_dataset_loader
from .dataset_loader.register import register_custom_dataset_loader
from .dataset_loader.register import register_json_dataset_loader
from .dataset_loader.register import register_jsonl_dataset_loader
from .dataset_loader.register import register_parquet_dataset_loader
from .dataset_loader.register import register_xls_dataset_loader
