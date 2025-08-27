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

from pydantic import model_validator


@dataclass
class GatedFieldMixinConfig:
    """Configuration for a gated field mixin."""

    field_name: str
    default_if_supported: object | None
    unsupported: Sequence[Pattern[str]] | None
    supported: Sequence[Pattern[str]] | None
    keys: Sequence[str]


class GatedFieldMixin:
    """
    A mixin that gates a field based on specified keys.

    This should be used to automatically validate a field based on a given key.

    Parameters
    ----------
    field_name: `str`
                The name of the field.
    default_if_supported: `object | None`
                          The default value of the field if it is supported for the key.
    keys: `Sequence[str]`
          A sequence of keys that are used to validate the field.
    unsupported: `Sequence[Pattern[str]] | None`
                 A sequence of regex patterns that match the key names NOT supported for the field.
                 Defaults to None.
    supported: `Sequence[Pattern[str]] | None`
               A sequence of regex patterns that match the key names supported for the field.
               Defaults to None.
    """

    def __init_subclass__(
        cls,
        field_name: str | None = None,
        default_if_supported: object | None = None,
        keys: Sequence[str] | None = None,
        unsupported: Sequence[Pattern[str]] | None = None,
        supported: Sequence[Pattern[str]] | None = None,
    ) -> None:
        """Store the class variables for the field and define the gated field validator."""
        super().__init_subclass__()

        # Check if this class directly inherits from GatedFieldMixin
        has_gated_field_mixin = GatedFieldMixin in cls.__bases__

        if has_gated_field_mixin:
            if keys is None:
                raise ValueError("keys must be provided when subclassing GatedFieldMixin")
            if field_name is None:
                raise ValueError("field_name must be provided when subclassing GatedFieldMixin")

            cls._setup_direct_mixin(field_name, default_if_supported, unsupported, supported, keys)

        # Always try to collect mixins and create validators for multiple inheritance
        # This handles both direct inheritance and deep inheritance chains
        all_mixins = cls._collect_all_mixin_configs()
        if all_mixins:
            cls._create_combined_validator(all_mixins)

    @classmethod
    def _setup_direct_mixin(
        cls,
        field_name: str,
        default_if_supported: object | None,
        unsupported: Sequence[Pattern[str]] | None,
        supported: Sequence[Pattern[str]] | None,
        keys: Sequence[str],
    ) -> None:
        """Set up a class that directly inherits from GatedFieldMixin."""
        cls._validate_mixin_parameters(unsupported, supported, keys)

        # Create and store validator
        validator = cls._create_gated_field_validator(field_name, default_if_supported, unsupported, supported, keys)
        validator_name = f"_gated_field_validator_{field_name}"
        setattr(cls, validator_name, validator)

        # Store mixin info for multiple inheritance
        if not hasattr(cls, "_gated_field_mixins"):
            cls._gated_field_mixins = []

        cls._gated_field_mixins.append(
            GatedFieldMixinConfig(
                field_name,
                default_if_supported,
                unsupported,
                supported,
                keys,
            ))

    @classmethod
    def _validate_mixin_parameters(
        cls,
        unsupported: Sequence[Pattern[str]] | None,
        supported: Sequence[Pattern[str]] | None,
        keys: Sequence[str],
    ) -> None:
        """Validate that all required parameters are provided."""
        if unsupported is None and supported is None:
            raise ValueError("Either unsupported or supported must be provided")
        if unsupported is not None and supported is not None:
            raise ValueError("Only one of unsupported or supported must be provided")
        if len(keys) == 0:
            raise ValueError("keys must be provided and non-empty when subclassing GatedFieldMixin")

    @classmethod
    def _create_gated_field_validator(
        cls,
        field_name: str,
        default_if_supported: object | None,
        unsupported: Sequence[Pattern[str]] | None,
        supported: Sequence[Pattern[str]] | None,
        keys: Sequence[str],
    ):
        """Create the model validator function."""

        @model_validator(mode="after")
        def gated_field_validator(self):
            """Validate the gated field."""
            current_value = getattr(self, field_name, None)
            is_supported = cls._check_field_support(self, unsupported, supported, keys)
            if not is_supported:
                if current_value is not None:
                    blocking_key = cls._find_blocking_key(self, unsupported, supported, keys)
                    value = getattr(self, blocking_key, "<unknown>")
                    raise ValueError(f"{field_name} is not supported for {blocking_key}: {value}")
            elif current_value is None:
                setattr(self, field_name, default_if_supported)
            return self

        return gated_field_validator

    @classmethod
    def _check_field_support(
        cls,
        instance: object,
        unsupported: Sequence[Pattern[str]] | None,
        supported: Sequence[Pattern[str]] | None,
        keys: Sequence[str],
    ) -> bool:
        """Check if a specific field is supported based on its configuration and keys."""
        seen = False
        for key in keys:
            if not hasattr(instance, key):
                continue
            seen = True
            value = str(getattr(instance, key))
            if supported is not None:
                if any(p.search(value) for p in supported):
                    return True
            elif unsupported is not None:
                if any(p.search(value) for p in unsupported):
                    return False
        return True if not seen else (unsupported is not None)

    @classmethod
    def _find_blocking_key(
        cls,
        instance: object,
        unsupported: Sequence[Pattern[str]] | None,
        supported: Sequence[Pattern[str]] | None,
        keys: Sequence[str],
    ) -> str:
        """Find which key is blocking the field."""
        for key in keys:
            if not hasattr(instance, key):
                continue
            value = str(getattr(instance, key))
            if supported is not None:
                if not any(p.search(value) for p in supported):
                    return key
            elif unsupported is not None:
                if any(p.search(value) for p in unsupported):
                    return key

        return "<unknown>"

    @classmethod
    def _collect_all_mixin_configs(cls) -> list[GatedFieldMixinConfig]:
        """Collect all mixin configurations from base classes."""
        all_mixins = []
        for base in cls.__bases__:
            if hasattr(base, "_gated_field_mixins"):
                all_mixins.extend(base._gated_field_mixins)
        return all_mixins

    @classmethod
    def _create_combined_validator(cls, all_mixins: list[GatedFieldMixinConfig]) -> None:
        """Create a combined validator that handles all fields."""

        @model_validator(mode="after")
        def combined_gated_field_validator(self):
            """Validate all gated fields."""
            for mixin_config in all_mixins:
                field_name_local = mixin_config.field_name
                current_value = getattr(self, field_name_local, None)
                if not self._check_field_support_instance(mixin_config):
                    if current_value is not None:
                        blocking_key = self._find_blocking_key_instance(mixin_config)
                        value = getattr(self, blocking_key, "<unknown>")
                        raise ValueError(f"{field_name_local} is not supported for {blocking_key}: {value}")
                elif current_value is None:
                    setattr(self, field_name_local, mixin_config.default_if_supported)

            return self

        cls._combined_gated_field_validator = combined_gated_field_validator

        # Add helper methods
        def _check_field_support_instance(self, mixin_config: GatedFieldMixinConfig) -> bool:
            """Check if a specific field is supported based on its configuration and keys."""
            return cls._check_field_support(self, mixin_config.unsupported, mixin_config.supported, mixin_config.keys)

        def _find_blocking_key_instance(self, mixin_config: GatedFieldMixinConfig) -> str:
            """Find which key is blocking the field."""
            return cls._find_blocking_key(self, mixin_config.unsupported, mixin_config.supported, mixin_config.keys)

        cls._check_field_support_instance = _check_field_support_instance
        cls._find_blocking_key_instance = _find_blocking_key_instance
