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
"""
Compatibility re-exports for YAML-backed evaluation config models.
This file can be dropped in NAT 1.6.0.
"""

import warnings

from nat.data_models.evaluate_config import EvalConfig  # noqa: F401
from nat.data_models.evaluate_config import EvalCustomScriptConfig  # noqa: F401
from nat.data_models.evaluate_config import EvalGeneralConfig  # noqa: F401
from nat.data_models.evaluate_config import EvalOutputConfig  # noqa: F401
from nat.data_models.evaluate_config import JobEvictionPolicy  # noqa: F401
from nat.data_models.evaluate_config import JobManagementConfig  # noqa: F401

warnings.warn(
    "Importing from 'nat.data_models.evaluate' is deprecated. "
    "Use 'nat.data_models.evaluate_config' for eval config models and "
    "'nat.data_models.evaluate_runtime' for runtime models.",
    UserWarning,
    stacklevel=2,
)
