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
"""Base class for AIQ Toolkit functions providing type handling and schema management.

This module contains the FunctionBase abstract base class which provides core functionality
for AIQ Toolkit functions including type handling via generics, schema management for inputs and outputs,
and type conversion capabilities.
"""

import logging
import typing
from abc import ABC
from collections.abc import Callable
from functools import lru_cache
from types import NoneType

from pydantic import BaseModel

from aiq.utils.type_converter import TypeConverter
from aiq.utils.type_utils import DecomposedType

InputT = typing.TypeVar("InputT")
StreamingOutputT = typing.TypeVar("StreamingOutputT")
SingleOutputT = typing.TypeVar("SingleOutputT")

logger = logging.getLogger(__name__)


class FunctionBase(typing.Generic[InputT, StreamingOutputT, SingleOutputT], ABC):
    """
    Abstract base class providing core functionality for AIQ Toolkit functions.

    This class provides type handling via generics, schema management for inputs and outputs,
    and type conversion capabilities.

    Parameters
    ----------
    InputT : TypeVar
        The input type for the function
    StreamingOutputT : TypeVar
        The output type for streaming results
    SingleOutputT : TypeVar
        The output type for single results

    Notes
    -----
    FunctionBase is the foundation of the AIQ Toolkit function system, providing:
    - Type handling via generics
    - Schema management for inputs and outputs
    - Type conversion capabilities
    - Abstract interface that concrete function classes must implement
    """

    def __init__(self,
                 *,
                 input_schema: type[BaseModel] | None = None,
                 streaming_output_schema: type[BaseModel] | type[None] | None = None,
                 single_output_schema: type[BaseModel] | type[None] | None = None,
                 converters: list[Callable[[typing.Any], typing.Any]] | None = None):

        converters = converters or []

        self._converter_list = converters

        final_input_schema = input_schema or DecomposedType(self.input_type).get_pydantic_schema(converters)

        assert not issubclass(final_input_schema, NoneType)

        self._input_schema = final_input_schema

        if streaming_output_schema is not None:
            self._streaming_output_schema = streaming_output_schema
        else:
            self._streaming_output_schema = DecomposedType(self.streaming_output_type).get_pydantic_schema(converters)

        if single_output_schema is not None:
            self._single_output_schema = single_output_schema
        else:
            self._single_output_schema = DecomposedType(self.single_output_type).get_pydantic_schema(converters)

        self._converter: TypeConverter = TypeConverter(converters)

    @property
    @lru_cache
    def input_type(self) -> type[InputT]:
        """
        Get the input type of the function. The input type is determined by the generic parameters of the class.

        For example, if a function is defined as `def my_function(input: list[int]) -> str`, the `input_type` is
        `list[int]`.

        Returns
        -------
        type[InputT]
            The input type specified in the generic parameters

        Raises
        ------
        ValueError
            If the input type cannot be determined from the class definition
        """
        for base_cls in self.__class__.__orig_bases__:  # pylint: disable=no-member # type: ignore

            base_cls_args = typing.get_args(base_cls)

            if len(base_cls_args) == 3:
                return base_cls_args[0]

        raise ValueError("Could not find input schema")

    @property
    @lru_cache
    def input_class(self) -> type:
        """
        Get the python class of the input type. This is the class that can be used to check if a value is an instance of
        the input type. It removes any generic or annotation information from the input type.

        For example, if a function is defined as `def my_function(input: list[int]) -> str`, the `input_class` is
        `list`.

        Returns
        -------
        type
            The python type of the input type
        """

        input_origin = typing.get_origin(self.input_type)

        if (input_origin is None):
            return self.input_type

        return input_origin

    @property
    @lru_cache
    def input_schema(self) -> type[BaseModel]:
        """
        Get the Pydantic model schema for validating inputs. The schema must be pydantic models. This allows for
        type validation and coercion, and documenting schema properties of the input value. If the input type is
        already a pydantic model, it will be returned as is.

        For example, if a function is defined as `def my_function(input: list[int]) -> str`, the `input_schema` is::

            class InputSchema(BaseModel):
                input: list[int]


        Returns
        -------
        type[BaseModel]
            The Pydantic model class for input validation
        """
        return self._input_schema

    @property
    def converter_list(self) -> list[Callable[[typing.Any], typing.Any]]:
        """
        Get the list of type converters used by this function.

        Returns
        -------
        list[Callable[[typing.Any], typing.Any]]
            List of converter functions that transform input types
        """
        return self._converter_list

    @property
    @lru_cache
    def streaming_output_type(self) -> type[StreamingOutputT]:
        """
        Get the streaming output type of the function. The streaming output type is determined by the generic parameters
        of the class.

        For example, if a function is defined as `def my_function(input: int) -> AsyncGenerator[dict[str, Any]]`,
        the `streaming_output_type` is `dict[str, Any]`.

        Returns
        -------
        type[StreamingOutputT]
            The streaming output type specified in the generic parameters

        Raises
        ------
        ValueError
            If the streaming output type cannot be determined from the class definition
        """
        for base_cls in self.__class__.__orig_bases__:  # pylint: disable=no-member # type: ignore

            base_cls_args = typing.get_args(base_cls)

            if len(base_cls_args) == 3:
                return base_cls_args[1]

        raise ValueError("Could not find output schema")

    @property
    @lru_cache
    def streaming_output_class(self) -> type:
        """
        Get the python class of the output type. This is the class that can be used to check if a value is an instance
        of the output type. It removes any generic or annotation information from the output type.

        For example, if a function is defined as `def my_function(input: int) -> AsyncGenerator[dict[str, Any]]`,
        the `streaming_output_class` is `dict`.

        Returns
        -------
        type
            The python type of the output type
        """

        output_origin = typing.get_origin(self.streaming_output_type)

        if (output_origin is None):
            return self.streaming_output_type

        return output_origin

    @property
    @lru_cache
    def streaming_output_schema(self) -> type[BaseModel] | type[None]:
        """
        Get the Pydantic model schema for validating streaming outputs. The schema must be pydantic models. This allows
        for type validation and coercion, and documenting schema properties of the output value. If the output type is
        already a pydantic model, it will be returned as is.

        For example, if a function is defined as `def my_function(input: int) -> AsyncGenerator[dict[str, Any]]`,
        the `streaming_output_schema` is::

            class StreamingOutputSchema(BaseModel):
                value: dict[str, Any]

        Returns
        -------
        type[BaseModel] | type[None]
            The Pydantic model class for streaming output validation, or NoneType if no streaming output.
        """
        return self._streaming_output_schema

    @property
    @lru_cache
    def single_output_type(self) -> type[SingleOutputT]:
        """
        Get the single output type of the function. The single output type is determined by the generic parameters
        of the class. Returns NoneType if no single output is supported.

        For example, if a function is defined as `def my_function(input: int) -> list[str]`, the `single_output_type` is
        `list[str]`.

        Returns
        -------
        type[SingleOutputT]
            The single output type specified in the generic parameters

        Raises
        ------
        ValueError
            If the single output type cannot be determined from the class definition
        """
        for base_cls in self.__class__.__orig_bases__:  # pylint: disable=no-member # type: ignore

            base_cls_args = typing.get_args(base_cls)

            if len(base_cls_args) == 3:
                return base_cls_args[2]

        raise ValueError("Could not find output schema")

    @property
    @lru_cache
    def single_output_class(self) -> type:
        """
        Get the python class of the output type. This is the class that can be used to check if a value is an instance
        of the output type. It removes any generic or annotation information from the output type.

        For example, if a function is defined as `def my_function(input: int) -> list[str]`, the `single_output_class`
        is `list`.

        Returns
        -------
        type
            The python type of the output type
        """

        output_origin = typing.get_origin(self.single_output_type)

        if (output_origin is None):
            return self.single_output_type

        return output_origin

    @property
    @lru_cache
    def single_output_schema(self) -> type[BaseModel] | type[None]:
        """
        Get the Pydantic model schema for validating single outputs. The schema must be pydantic models. This allows for
        type validation and coercion, and documenting schema properties of the output value. If the output type is
        already a pydantic model, it will be returned as is.

        For example, if a function is defined as `def my_function(input: int) -> list[str]`, the `single_output_schema`
        is::

            class SingleOutputSchema(BaseModel):
                value: list[str]

        Returns
        -------
        type[BaseModel] | type[None]
            The Pydantic model class for single output validation, or None if no single output
        """
        return self._single_output_schema

    @property
    def has_streaming_output(self) -> bool:
        """
        Check if this function supports streaming output.

        Returns
        -------
        bool
            True if the function supports streaming output, False otherwise
        """
        # Override in derived classes if this needs to return False. Assumption is, if not overridden, it has streaming
        # output because the ABC has it.
        return True

    @property
    def has_single_output(self) -> bool:
        """
        Check if this function supports single output.

        Returns
        -------
        bool
            True if the function supports single output, False otherwise
        """
        # Override in derived classes if this needs to return False. Assumption is, if not overridden, it has single
        # output because the ABC has it.
        return True

    def _convert_input(self, value: typing.Any):
        if (isinstance(value, self.input_class)):
            return value

        # No converter, try to convert to the input schema
        if (isinstance(value, dict)):
            value = self.input_schema.model_validate(value)

            if (self.input_type == self.input_schema):
                return value

        if (isinstance(value, self.input_schema)):

            # Get the first value from the schema object
            first_key = next(iter(self.input_schema.model_fields.keys()))

            return getattr(value, first_key)

        # If the value is None bypass conversion to avoid raising an error.
        if value is None:
            return value

        # Fallback to the converter
        return self._converter.convert(value, to_type=self.input_class)
