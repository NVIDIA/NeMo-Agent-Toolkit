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
from typing import ClassVar
from typing import Generic
from typing import TypeVar

from pydantic import BaseModel
from pydantic import model_validator

T = TypeVar("T")


class SupportedField(Generic[T]):
    """
    A supported field is a field that is supported (or not supported) for a given model.

    This should be used to simplify the validation of a field for a given model.

    Args:
        field_name: The name of the field.
        default_if_supported: The default value of the field if it is supported for the model.
        unsupported_models: A sequence of regex patterns that match the model names NOT supported for the field.
        supported_models: A sequence of regex patterns that match the model names supported for the field.
        model_keys: A sequence of keys that are used to validate the field.
                    Defaults to ("model_name", "model", "azure_deployment",)
    """

    # The keys that are used to validate the field. This should be extended to support additional keys
    # when needed, but should not be modified by subclasses. If subclasses need to add additional keys,
    # they should override the model_keys class variable.
    model_keys: ClassVar[Sequence[str]] = (
        "model_name",
        "model",
        "azure_deployment",
    )

    def __init_subclass__(
        cls,
        field_name: str | None = None,
        default_if_supported: T | None = None,
        unsupported_models: Sequence[Pattern[str]] | None = None,
        supported_models: Sequence[Pattern[str]] | None = None,
        model_keys: Sequence[str] | None = None,
    ) -> None:
        """
        Store the class variables for the field and define the model validator.
        """
        super().__init_subclass__()
        if SupportedField in cls.__bases__:
            if field_name is None:
                raise ValueError("field_name must be provided when subclassing SupportedField")
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
                    unsupported: A sequence of regex patterns that match the model names NOT supported for the field.
                """
                if cls.unsupported_models is not None:
                    return not any(p.search(model_name) for p in cls.unsupported_models)
                if cls.supported_models is not None:
                    return any(p.search(model_name) for p in cls.supported_models)
                return False

            cls._supported_field_check_model = check_model

            @classmethod
            def configuration_valid(cls) -> None:
                if getattr(cls, "unsupported_models", None) is None and getattr(cls, "supported_models", None) is None:
                    raise ValueError("Either unsupported_models or supported_models must be provided")
                if getattr(cls, "unsupported_models", None) is not None and getattr(cls, "supported_models",
                                                                                    None) is not None:
                    raise ValueError("Only one of unsupported_models or supported_models must be provided")

            cls._supported_field_ensure_selector_configuration_valid = configuration_valid

            @classmethod
            def resolve_support(cls, instance: BaseModel) -> tuple[str, bool] | None:
                for key in cls.model_keys:
                    if hasattr(instance, key):
                        model_name_value = getattr(instance, key)
                        is_supported = cls._supported_field_check_model(str(model_name_value))
                        return key, is_supported
                return None

            cls._supported_field_resolve_support = resolve_support

            @model_validator(mode="after")
            def model_validate(self):
                clss = self.__class__
                clss._supported_field_ensure_selector_configuration_valid()

                field_name_local = getattr(clss, "field_name")
                current_value = getattr(self, field_name_local, None)

                match clss._supported_field_resolve_support(self):
                    case (found_key, is_supported):
                        if not is_supported:
                            if current_value is not None:
                                raise ValueError(
                                    f"{field_name_local} is not supported for {found_key}: {getattr(self, found_key)}")
                            return self
                    case None:
                        # No model key present → apply default
                        if current_value is None:
                            setattr(self, field_name_local, getattr(clss, "default_if_supported", None))
                        return self

                if current_value is None:
                    setattr(self, field_name_local, getattr(clss, "default_if_supported", None))
                return self

            cls._supported_field_model_validator = model_validate
