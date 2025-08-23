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

import typing

from pydantic import Field

from .common import BaseModelRegistryTag
from .common import TypedBaseModel


class FunctionBaseConfig(TypedBaseModel, BaseModelRegistryTag):
    pass


class FunctionGroupBaseConfig(TypedBaseModel, BaseModelRegistryTag):
    """Base configuration for function groups.

    Function groups enable sharing of configurations and resources across multiple functions.
    """
    exposes: list[str] = Field(
        default=[],
        description="The list of exposed function names which should be added to the global Function registry",
    )


class EmptyFunctionConfig(FunctionBaseConfig, name="EmptyFunctionConfig"):
    pass


FunctionConfigT = typing.TypeVar("FunctionConfigT", bound=FunctionBaseConfig)

FunctionGroupConfigT = typing.TypeVar("FunctionGroupConfigT", bound=FunctionGroupBaseConfig)
