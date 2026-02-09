# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
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

"""Microsoft Agent 365 front-end plugin."""

from nat.plugins.a365.exceptions import (
    A365AuthenticationError,
    A365ConfigurationError,
    A365Error,
    A365SDKError,
    A365WorkflowExecutionError,
)

from .front_end_config import A365FrontEndConfig
from .plugin import A365FrontEndPlugin
from .worker import A365FrontEndPluginWorker

__all__ = [
    "A365FrontEndConfig",
    "A365FrontEndPlugin",
    "A365FrontEndPluginWorker",
    "A365Error",
    "A365AuthenticationError",
    "A365ConfigurationError",
    "A365WorkflowExecutionError",
    "A365SDKError",
]
