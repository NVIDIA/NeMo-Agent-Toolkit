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
"""Compatibility re-export for token usage model from core."""

import warnings

from nat.data_models.token_usage import TokenUsageBaseModel  # noqa: F401  # pyright: ignore[reportMissingImports]

warnings.warn(
    "Importing TokenUsageBaseModel from 'nat.plugins.profiler.callbacks.token_usage_base_model' is deprecated. "
    "Use 'nat.data_models.token_usage.TokenUsageBaseModel' instead.",
    UserWarning,
    stacklevel=2,
)
