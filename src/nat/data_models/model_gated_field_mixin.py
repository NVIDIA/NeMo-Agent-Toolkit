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
from dataclasses import dataclass
from re import Pattern
from typing import Generic
from typing import TypeVar

from pydantic import BaseModel
from pydantic import model_validator


@dataclass
class ModelGatedFieldMixinConfig:
    """Configuration for a model-gated field mixin."""

    field_name: str
    default_if_supported: object | None
    unsupported_models: Sequence[Pattern[str]] | None
    supported_models: Sequence[Pattern[str]] | None
    model_keys: Sequence[str]


T = TypeVar("T")


class ModelGatedFieldMixin(Generic[T]):
    """
    A mixin that gates a field based on model support.

    This should be used to automatically validate a field based on a given model.

    Parameters
    ----------
    field_name: `str`
                The name of the field.
    default_if_supported: `T | None`
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
        """Store the class variables for the field and define the model validator."""
        super().__init_subclass__()

        has_model_gated_mixin = ModelGatedFieldMixin in cls.__bases__

        if has_model_gated_mixin:
            if field_name is None:
                raise ValueError("field_name must be provided when subclassing ModelGatedFieldMixin")
            cls._setup_direct_mixin(field_name, default_if_supported, unsupported_models, supported_models, model_keys)

        # Always try to collect mixins and create validators for multiple inheritance
        # This handles both direct inheritance and deep inheritance chains
        all_mixins = cls._collect_all_mixin_configs()
        if all_mixins:
            cls._create_combined_validator(all_mixins)

    @classmethod
    def _setup_direct_mixin(
        cls,
        field_name: str,
        default_if_supported: T | None,
        unsupported_models: Sequence[Pattern[str]] | None,
        supported_models: Sequence[Pattern[str]] | None,
        model_keys: Sequence[str],
    ) -> None:
        """Set up a class that directly inherits from ModelGatedFieldMixin."""
        cls._validate_mixin_parameters(unsupported_models, supported_models, model_keys)

        # Create and store validator
        validator = cls._create_model_validator(field_name,
                                                default_if_supported,
                                                unsupported_models,
                                                supported_models,
                                                model_keys)
        validator_name = f"_model_gated_field_model_validator_{field_name}"
        setattr(cls, validator_name, validator)

        # Store mixin info for multiple inheritance
        if not hasattr(cls, "_model_gated_mixins"):
            cls._model_gated_mixins = []

        cls._model_gated_mixins.append(
            ModelGatedFieldMixinConfig(
                field_name,
                default_if_supported,
                unsupported_models,
                supported_models,
                model_keys,
            ))

    @classmethod
    def _validate_mixin_parameters(
        cls,
        unsupported_models: Sequence[Pattern[str]] | None,
        supported_models: Sequence[Pattern[str]] | None,
        model_keys: Sequence[str],
    ) -> None:
        """Validate that all required parameters are provided."""
        if unsupported_models is None and supported_models is None:
            raise ValueError("Either unsupported_models or supported_models must be provided")
        if unsupported_models is not None and supported_models is not None:
            raise ValueError("Only one of unsupported_models or supported_models must be provided")
        if model_keys is not None and len(model_keys) == 0:
            raise ValueError("model_keys must be provided and non-empty when subclassing ModelGatedFieldMixin")

    @classmethod
    def _create_model_validator(
        cls,
        field_name: str,
        default_if_supported: T | None,
        unsupported_models: Sequence[Pattern[str]] | None,
        supported_models: Sequence[Pattern[str]] | None,
        model_keys: Sequence[str],
    ):
        """Create the model validator function."""

        @model_validator(mode="after")
        def model_validate(self):
            """Validate the model-gated field."""
            current_value = getattr(self, field_name, None)
            is_supported = cls._check_field_support(self, unsupported_models, supported_models, model_keys)
            if not is_supported:
                if current_value is not None:
                    blocking_key = cls._find_blocking_key(self, unsupported_models, supported_models, model_keys)
                    model_value = getattr(self, blocking_key, "<unknown>")
                    raise ValueError(f"{field_name} is not supported for {blocking_key}: {model_value}")
            elif current_value is None:
                setattr(self, field_name, default_if_supported)
            return self

        return model_validate

    @classmethod
    def _check_field_support(
        cls,
        instance: BaseModel,
        unsupported_models: Sequence[Pattern[str]] | None,
        supported_models: Sequence[Pattern[str]] | None,
        model_keys: Sequence[str],
    ) -> bool:
        """Check if a specific field is supported based on its configuration."""
        for key in model_keys:
            if not hasattr(instance, key):
                continue
            model_value = str(getattr(instance, key))
            if supported_models is not None:
                return any(p.search(model_value) for p in supported_models)
            elif unsupported_models is not None:
                return not any(p.search(model_value) for p in unsupported_models)
        # Default to supported if no model keys found
        return True

    @classmethod
    def _find_blocking_key(
        cls,
        instance: BaseModel,
        unsupported_models: Sequence[Pattern[str]] | None,
        supported_models: Sequence[Pattern[str]] | None,
        model_keys: Sequence[str],
    ) -> str:
        """Find which model key is blocking the field."""
        for key in model_keys:
            if not hasattr(instance, key):
                continue
            model_value = str(getattr(instance, key))
            if supported_models is not None:
                if not any(p.search(model_value) for p in supported_models):
                    return key
            elif unsupported_models is not None:
                if any(p.search(model_value) for p in unsupported_models):
                    return key

        return "<unknown>"

    @classmethod
    def _collect_all_mixin_configs(cls) -> list[ModelGatedFieldMixinConfig]:
        """Collect all mixin configurations from base classes."""
        all_mixins = []
        for base in cls.__bases__:
            if hasattr(base, "_model_gated_mixins"):
                all_mixins.extend(base._model_gated_mixins)
        return all_mixins

    @classmethod
    def _create_combined_validator(cls, all_mixins: list[ModelGatedFieldMixinConfig]) -> None:
        """Create a combined validator that handles all fields."""

        @model_validator(mode="after")
        def combined_model_validate(self):
            """Validate all model-gated fields."""
            for mixin_config in all_mixins:
                field_name_local = mixin_config.field_name
                current_value = getattr(self, field_name_local, None)
                if not self._check_field_support_instance(mixin_config):
                    if current_value is not None:
                        blocking_key = self._find_blocking_key_instance(mixin_config)
                        model_value = getattr(self, blocking_key, "<unknown>")
                        raise ValueError(f"{field_name_local} is not supported for {blocking_key}: {model_value}")
                elif current_value is None:
                    setattr(self, field_name_local, mixin_config.default_if_supported)

            return self

        cls._combined_model_gated_field_validator = combined_model_validate

        # Add helper methods
        def _check_field_support_instance(self, mixin_config: ModelGatedFieldMixinConfig) -> bool:
            """Check if a specific field is supported based on its configuration."""
            return cls._check_field_support(self,
                                            mixin_config.unsupported_models,
                                            mixin_config.supported_models,
                                            mixin_config.model_keys)

        def _find_blocking_key_instance(self, mixin_config: ModelGatedFieldMixinConfig) -> str:
            """Find which model key is blocking the field."""
            return cls._find_blocking_key(self,
                                          mixin_config.unsupported_models,
                                          mixin_config.supported_models,
                                          mixin_config.model_keys)

        cls._check_field_support_instance = _check_field_support_instance
        cls._find_blocking_key_instance = _find_blocking_key_instance
