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

import logging
import typing
from collections import OrderedDict
from collections.abc import Callable
from io import TextIOWrapper

from aiq.utils.type_utils import DecomposedType

logger = logging.getLogger(__name__)

_T = typing.TypeVar("_T")


class ConvertException(Exception):
    pass


class TypeConverter:
    _global_initialized = False

    def __init__(self, converters: list[Callable[[typing.Any], typing.Any]], parent: "TypeConverter | None" = None):
        """
        Parameters
        ----------
        converters : list[Callable[[typing.Any], typing.Any]]
            A list of single-argument converter callables annotated with their input param and return type.
        parent : TypeConverter | None
            An optional parent TypeConverter for fallback.
        """
        # dict[to_type, dict[from_type, converter]]
        self._converters: OrderedDict[type, OrderedDict[type, Callable]] = OrderedDict()
        self._indirect_warnings_shown: set[tuple[type, type]] = set()

        for converter in converters:
            self.add_converter(converter)

        if parent is None and TypeConverter._global_initialized:
            parent = GlobalTypeConverter.get()
        self._parent = parent

    def add_converter(self, converter: Callable) -> None:
        """
        Registers a converter. Must have exactly one parameter
        and an annotated return type.

        Parameters
        ----------
        converter : Callable
            A converter function. Must have exactly one parameter and an annotated return type.

        Raises
        ------
        ValueError
            If the converter does not have a return type or exactly one argument or the argument has no data type.
        """
        sig = typing.get_type_hints(converter)
        to_type = sig.pop("return", None)
        if to_type is None:
            raise ValueError("Converter must have a return type.")

        if len(sig) != 1:
            raise ValueError("Converter must have exactly one argument.")

        from_type = next(iter(sig.values()))
        if from_type is None:
            raise ValueError("Converter's argument must have a data type.")

        self._converters.setdefault(to_type, OrderedDict())[from_type] = converter
        # to do(MDD): If needed, sort by specificity here.

    def _convert(self, data: typing.Any, to_type: type[_T]) -> _T | None:
        """
        Attempts to convert `data` into `to_type`. Returns None if no path is found.
        """
        decomposed = DecomposedType(to_type)

        # 1) If data is already correct type, return it
        if to_type is None or decomposed.is_instance((data, to_type)):
            return data

        root = decomposed.root

        # 2) Attempt direct in *this* converter
        direct_result = self._try_direct_conversion(data, root)
        if direct_result is not None:
            return direct_result

        # 3) If direct fails entirely, do indirect in *this* converter
        indirect_result = self._try_indirect_convert(data, to_type)
        if indirect_result is not None:
            return indirect_result

        # 4) If we still haven't succeeded, return None
        return None

    def convert(self, data: typing.Any, to_type: type[_T]) -> _T:
        """
        Converts or raises ValueError if no conversion path is found.
        We also give the parent a chance if self fails.

        Parameters
        ----------
        data : typing.Any
            The value to convert.
        to_type : type
            The type to convert the value to.

        Returns
        -------
        _T
            The converted value.

        Raises
        ------
        ValueError
            If the value cannot be converted to the specified type.
        """
        result = self._convert(data, to_type)
        if result is None and self._parent:
            # fallback on parent entirely
            return self._parent.convert(data, to_type)

        if result is not None:
            return result
        raise ValueError(f"Cannot convert type {type(data)} to {to_type}. No match found.")

    def try_convert(self, data: typing.Any, to_type: type[_T]) -> _T | typing.Any:
        """
        Converts with graceful error handling. If conversion fails, returns the original data
        and continues processing.

        Parameters
        ----------
        data : typing.Any
            The value to convert.
        to_type : type
            The type to convert the value to.

        Returns
        -------
        _T | typing.Any
            The converted value, or original value if conversion fails.
        """
        try:
            return self.convert(data, to_type)
        except ValueError:
            logger.warning("Type conversion failed, using original value. From %s to %s", type(data), to_type)
            # Return original data, let downstream code handle it
            return data

    # -------------------------------------------------
    # INTERNAL DIRECT CONVERSION (with parent fallback)
    # -------------------------------------------------
    def _try_direct_conversion(self, data: typing.Any, target_root_type: type) -> typing.Any | None:
        """
        Tries direct conversion in *this* converter's registry.
        If no match here, we forward to parent's direct conversion
        for recursion up the chain.
        """
        for convert_to_type, to_type_converters in self._converters.items():
            # e.g. if Derived is a subclass of Base, this is valid
            if issubclass(DecomposedType(convert_to_type).root, target_root_type):
                for convert_from_type, from_type_converter in to_type_converters.items():
                    if isinstance(data, DecomposedType(convert_from_type).root):
                        try:
                            return from_type_converter(data)
                        except ConvertException:
                            pass

        # If we can't convert directly here, try parent
        if self._parent is not None:
            return self._parent._try_direct_conversion(data, target_root_type)

        return None

    # -------------------------------------------------
    # INTERNAL INDIRECT CONVERSION (with parent fallback)
    # -------------------------------------------------
    def _try_indirect_convert(self, data: typing.Any, to_type: type[_T]) -> _T | None:
        """
        Attempt indirect conversion (DFS) in *this* converter.
        If no success, fallback to parent's indirect attempt.
        """
        visited = set()
        final = self._try_indirect_conversion(data, to_type, visited)
        if final is not None:
            # Warn once if found a chain
            self._maybe_warn_indirect(type(data), to_type)
            return final

        # If no success, try parent's indirect
        if self._parent is not None:
            parent_final = self._parent._try_indirect_convert(data, to_type)
            if parent_final is not None:
                self._maybe_warn_indirect(type(data), to_type)
                return parent_final

        return None

    def _try_indirect_conversion(self, data: typing.Any, to_type: type[_T], visited: set[type]) -> _T | None:
        """
        DFS attempt to find a chain of conversions from type(data) to to_type,
        ignoring parent. If not found, returns None.
        """
        # 1) If data is already correct type
        if isinstance(data, to_type):
            return data

        current_type = type(data)
        if current_type in visited:
            return None

        visited.add(current_type)

        # 2) Attempt each known converter from current_type -> ???, then recurse
        for _, to_type_converters in self._converters.items():
            for convert_from_type, from_type_converter in to_type_converters.items():
                if isinstance(data, convert_from_type):
                    try:
                        next_data = from_type_converter(data)
                        if isinstance(next_data, to_type):
                            return next_data
                        # else keep going
                        deeper = self._try_indirect_conversion(next_data, to_type, visited)
                        if deeper is not None:
                            return deeper
                    except ConvertException:
                        pass

        return None

    def _maybe_warn_indirect(self, source_type: type, to_type: type):
        """
        Warn once if an indirect path was used between these two types.
        """
        pair = (source_type, to_type)
        if pair not in self._indirect_warnings_shown:
            logger.warning(
                "Indirect type conversion used to convert %s to %s, which may lead to unintended conversions. "
                "Consider adding a direct converter from %s to %s to ensure correctness.",
                source_type,
                to_type,
                source_type,
                to_type)
            self._indirect_warnings_shown.add(pair)


class GlobalTypeConverter:
    _global_converter: TypeConverter = TypeConverter([])

    @staticmethod
    def get() -> TypeConverter:
        return GlobalTypeConverter._global_converter

    @staticmethod
    def register_converter(converter: Callable) -> None:
        GlobalTypeConverter._global_converter.add_converter(converter)

    @staticmethod
    def convert(data, to_type: type[_T]) -> _T:
        return GlobalTypeConverter._global_converter.convert(data, to_type)

    @staticmethod
    def try_convert(data: typing.Any, to_type: type[_T]) -> _T | typing.Any:
        return GlobalTypeConverter._global_converter.try_convert(data, to_type)


TypeConverter._global_initialized = True


def _text_io_wrapper_to_string(data: TextIOWrapper) -> str:
    return data.read()


GlobalTypeConverter.register_converter(_text_io_wrapper_to_string)
