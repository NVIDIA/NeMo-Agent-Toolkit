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

from collections.abc import Sequence
from re import Pattern
from typing import Generic
from typing import TypeVar

from pydantic import BaseModel
from pydantic import model_validator

T = TypeVar("T")


class ModelGatedFieldMixin(Generic[T]):
    """
    A mixin that gates a field based on model support.

    This should be used to automatically validate a field based on a given model.

    Parameters
    ----------
    field_name: `str`
                The name of the field.
    default_if_supported: `T`
                          The default value of the field if it is supported for the model.
    unsupported_models: `Sequence[Pattern[str]] | None`
                        A sequence of regex patterns that match the model names NOT supported for the field.
                        Defaults to None.
    supported_models: `Sequence[Pattern[str]] | None`
                      A sequence of regex patterns that match the model names supported for the field.
                      Defaults to None.
    model_keys: `Sequence[str]`
                A sequence of keys that are used to validate the field.
                Defaults to ("model_name", "model", "azure_deployment",)
    """

    def __init_subclass__(
            cls,
            field_name: str | None = None,
            default_if_supported: T | None = None,
            unsupported_models: Sequence[Pattern[str]] | None = None,
            supported_models: Sequence[Pattern[str]] | None = None,
            model_keys: Sequence[str] = ("model_name", "model", "azure_deployment"),
    ) -> None:
        """
        Store the class variables for the field and define the model validator.
        """
        super().__init_subclass__()
        if ModelGatedFieldMixin in cls.__bases__:
            if field_name is None:
                raise ValueError("field_name must be provided when subclassing ModelGatedFieldMixin")
            if default_if_supported is None:
                raise ValueError("default_if_supported must be provided when subclassing ModelGatedFieldMixin")
            if unsupported_models is None and supported_models is None:
                raise ValueError("Either unsupported_models or supported_models must be provided")
            if unsupported_models is not None and supported_models is not None:
                raise ValueError("Only one of unsupported_models or supported_models must be provided")
            if model_keys is not None and len(model_keys) == 0:
                raise ValueError("model_keys must be provided and non-empty when subclassing ModelGatedFieldMixin")
            cls.field_name = field_name
            cls.default_if_supported = default_if_supported
            cls.unsupported_models = unsupported_models
            cls.supported_models = supported_models
            if model_keys is not None:
                cls.model_keys = model_keys

            @classmethod
            def check_model(cls, model_name: str) -> bool:
                """
                Check if a model is supported for a given field.

                Args:
                    model_name: The name of the model to check.
                """
                unsupported = getattr(cls, "unsupported_models", None)
                supported = getattr(cls, "supported_models", None)
                if unsupported is not None:
                    return not any(p.search(model_name) for p in unsupported)
                if supported is not None:
                    return any(p.search(model_name) for p in supported)
                return False

            cls._model_gated_field_check_model = check_model

            @classmethod
            def detect_support(cls, instance: BaseModel) -> str | None:
                for key in getattr(cls, "model_keys"):
                    if hasattr(instance, key):
                        model_name_value = getattr(instance, key)
                        is_supported = getattr(cls, "_model_gated_field_check_model")(str(model_name_value))
                        return key if not is_supported else None
                return None

            cls._model_gated_field_detect_support = detect_support

            @model_validator(mode="after")
            def model_validate(self):
                klass = self.__class__

                field_name_local = getattr(klass, "field_name")
                current_value = getattr(self, field_name_local, None)

                found_key = klass._model_gated_field_detect_support(self)
                if found_key is not None:
                    if current_value is not None:
                        raise ValueError(
                            f"{field_name_local} is not supported for {found_key}: {getattr(self, found_key)}")
                elif current_value is None:
                    setattr(self, field_name_local, getattr(klass, "default_if_supported", None))
                return self

            cls._model_gated_field_model_validator = model_validate
