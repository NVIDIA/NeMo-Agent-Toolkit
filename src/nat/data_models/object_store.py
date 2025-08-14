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

import typing

from .common import BaseModelRegistryTag
from .common import TypedBaseModel


class ObjectStoreBaseConfig(TypedBaseModel, BaseModelRegistryTag):
    pass


ObjectStoreBaseConfigT = typing.TypeVar("ObjectStoreBaseConfigT", bound=ObjectStoreBaseConfig)


class KeyAlreadyExistsError(Exception):

    def __init__(self, key: str, additional_message: str | None = None):
        parts = [f"Key already exists: {key}."]
        if additional_message:
            parts.append(additional_message)
        super().__init__(" ".join(parts))


class NoSuchKeyError(Exception):

    def __init__(self, key: str, additional_message: str | None = None):
        parts = [f"No object found with key: {key}."]
        if additional_message:
            parts.append(additional_message)
        super().__init__(" ".join(parts))
