# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import inspect
import sys
import typing
from hashlib import sha512

from pydantic import AliasChoices
from pydantic import BaseModel
from pydantic import Field

_LT = typing.TypeVar("_LT")


class HashableBaseModel(BaseModel):
    """
    Subclass of a Pydantic BaseModel that is hashable. Use in objects that need to be hashed for caching purposes.
    """

    def __hash__(self):
        return int.from_bytes(bytes=sha512(f"{self.__class__.__qualname__}::{self.model_dump_json()}".encode(
            'utf-8', errors='ignore')).digest(),
                              byteorder=sys.byteorder)

    def __lt__(self, other):
        return self.__hash__() < other.__hash__()

    def __eq__(self, other):
        return self.__hash__() == other.__hash__()

    def __ne__(self, other):
        return self.__hash__() != other.__hash__()

    def __gt__(self, other):
        return self.__hash__() > other.__hash__()

    @classmethod
    def generate_json_schema(cls) -> dict[str, typing.Any]:
        return cls.model_json_schema()

    @classmethod
    def write_json_schema(cls, schema_path: str) -> None:

        import json

        schema = cls.generate_json_schema()

        with open(schema_path, "w", encoding="utf-8") as f:
            json.dump(schema, f, indent=2)


def subclass_depth(cls: type) -> int:
    """
    Compute a class' subclass depth.
    """
    depth = 0
    while (cls is not object):
        cls = cls.__base__
        depth += 1
    return depth


def _get_origin_or_base(cls: type) -> type:
    """
    Get the origin of a type or the base class if it is not a generic.
    """
    origin = typing.get_origin(cls)
    if origin is None:
        return cls
    return origin


class BaseModelRegistryTag:

    pass


class TypedBaseModel(BaseModel):
    """
    Subclass of Pydantic BaseModel that allows for specifying the object type. Use in Pydantic discriminated unions.
    """

    type: str = Field(init=False,
                      serialization_alias="_type",
                      validation_alias=AliasChoices('type', '_type'),
                      description="The type of the object",
                      title="Type",
                      repr=False)

    full_type: typing.ClassVar[str]

    def __init_subclass__(cls, name: str | None = None):
        super().__init_subclass__()

        if (name is not None):
            module = inspect.getmodule(cls)

            assert module is not None, f"Module not found for class {cls} when registering {name}"
            package_name: str | None = module.__package__

            # If the package name is not set, then we use the module name. Must have some namespace which will be unique
            if (not package_name):
                package_name = module.__name__

            full_name = f"{package_name}/{name}"

            type_field = cls.model_fields.get("type")
            if type_field is not None:
                type_field.default = name
            cls.full_type = full_name

    @classmethod
    def static_type(cls):
        return cls.model_fields.get("type").default

    @classmethod
    def static_full_type(cls):
        return cls.full_type

    @staticmethod
    def discriminator(v: typing.Any) -> str | None:
        # If its serialized, then we use the alias
        if isinstance(v, dict):
            return v.get("_type", v.get("type"))

        # Otherwise we use the property
        return getattr(v, "type")


TypedBaseModelT = typing.TypeVar("TypedBaseModelT", bound=TypedBaseModel)
