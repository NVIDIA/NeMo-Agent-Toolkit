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
from pydantic import field_validator
from pydantic import model_validator

from .common import BaseModelRegistryTag
from .common import TypedBaseModel


class FunctionBaseConfig(TypedBaseModel, BaseModelRegistryTag):
    """Base configuration for functions.

    Attributes:
        middleware: List of function middleware names to apply to this function.
            These must match names defined in the `middleware` section of the YAML configuration.
    """
    middleware: list[str] = Field(
        default_factory=list,
        description="List of function middleware names to apply to this function in order",
    )


class FunctionGroupBaseConfig(TypedBaseModel, BaseModelRegistryTag):
    """Base configuration for function groups.

    Function groups enable sharing of configurations and resources across multiple functions.
    """
    include: list[str] = Field(
        default_factory=list,
        description="The list of function names which should be added to the global Function registry",
    )
    exclude: list[str] = Field(
        default_factory=list,
        description="The list of function names which should be excluded from default access to the group",
    )
    middleware: list[str] = Field(
        default_factory=list,
        description="List of function middleware names to apply to all functions in this group",
    )

    @field_validator("include", "exclude")
    @classmethod
    def _validate_fields_include_exclude(cls, value: list[str]) -> list[str]:
        if len(set(value)) != len(value):
            raise ValueError("Function names must be unique")
        return sorted(value)

    @model_validator(mode="after")
    def _validate_include_exclude(self):
        if self.include and self.exclude:
            raise ValueError("include and exclude cannot be used together")
        return self


class EmptyFunctionConfig(FunctionBaseConfig, name="EmptyFunctionConfig"):
    pass


FunctionConfigT = typing.TypeVar("FunctionConfigT", bound=FunctionBaseConfig)

FunctionGroupConfigT = typing.TypeVar("FunctionGroupConfigT", bound=FunctionGroupBaseConfig)
