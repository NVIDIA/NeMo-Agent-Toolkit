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

from typing import Generic
from typing import TypeVar

from pydantic import BaseModel
from pydantic import Field

FrameworkT = TypeVar("FrameworkT")
ProviderT = TypeVar("ProviderT")


class TraceSourceBase(BaseModel, Generic[FrameworkT, ProviderT]):
    """Base class for trace sources with generic framework and provider types."""
    framework: FrameworkT = Field(..., description="The framework of the trace source")
    provider: ProviderT = Field(..., description="The provider of the trace source")
