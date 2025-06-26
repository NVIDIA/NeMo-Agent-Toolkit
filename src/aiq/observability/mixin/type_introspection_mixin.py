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

from functools import lru_cache
from typing import Any
from typing import get_args
from typing import get_origin


class TypeIntrospectionMixin:
    """Mixin class providing type introspection capabilities for generic classes.

    This mixin extracts type information from generic class definitions,
    allowing classes to determine their InputT and OutputT types at runtime.
    """

    @property
    @lru_cache
    def input_type(self) -> type[Any]:
        """
        Get the input type of the class. The input type is determined by the generic parameters of the class.

        For example, if a class is defined as `MyClass[list[int], str]`, the `input_type` is `list[int]`.

        Returns
        -------
        type[Any]
            The input type specified in the generic parameters

        Raises
        ------
        ValueError
            If the input type cannot be determined from the class definition
        """
        for base_cls in self.__class__.__orig_bases__:  # pylint: disable=no-member # type: ignore
            base_cls_args = get_args(base_cls)

            if len(base_cls_args) >= 2:
                return base_cls_args[0]

        raise ValueError(f"Could not find input type for {self.__class__.__name__}")

    @property
    @lru_cache
    def output_type(self) -> type[Any]:
        """
        Get the output type of the class. The output type is determined by the generic parameters of the class.

        For example, if a class is defined as `MyClass[list[int], str]`, the `output_type` is `str`.

        Returns
        -------
        type[Any]
            The output type specified in the generic parameters

        Raises
        ------
        ValueError
            If the output type cannot be determined from the class definition
        """
        for base_cls in self.__class__.__orig_bases__:  # pylint: disable=no-member # type: ignore
            base_cls_args = get_args(base_cls)

            if len(base_cls_args) >= 2:
                return base_cls_args[1]

        raise ValueError(f"Could not find output type for {self.__class__.__name__}")

    @property
    @lru_cache
    def input_class(self) -> type:
        """
        Get the python class of the input type. This is the class that can be used to check if a value is an
        instance of the input type. It removes any generic or annotation information from the input type.

        For example, if the input type is `list[int]`, the `input_class` is `list`.

        Returns
        -------
        type
            The python type of the input type
        """
        input_origin = get_origin(self.input_type)

        if input_origin is None:
            return self.input_type

        return input_origin

    @property
    @lru_cache
    def output_class(self) -> type:
        """
        Get the python class of the output type. This is the class that can be used to check if a value is an
        instance of the output type. It removes any generic or annotation information from the output type.

        For example, if the output type is `list[int]`, the `output_class` is `list`.

        Returns
        -------
        type
            The python type of the output type
        """
        output_origin = get_origin(self.output_type)

        if output_origin is None:
            return self.output_type

        return output_origin
