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

import re

import pytest
from pydantic import BaseModel
from pydantic import Field
from pydantic import ValidationError

from nat.data_models.model_gated_field_mixin import ModelGatedFieldMixin


def test_model_gated_field_requires_one_selector():

    with pytest.raises(ValueError, match=r"Only one of unsupported_models or supported_models must be provided"):

        class BadBoth(
                BaseModel,
                ModelGatedFieldMixin[int],
                field_name="dummy",
                default_if_supported=1,
                unsupported_models=(re.compile(r"alpha"), ),
                supported_models=(re.compile(r"beta"), ),
        ):
            dummy: int | None = Field(default=None)
            model_name: str = "alpha"

        _ = BadBoth()


def test_model_gated_field_requires_selector_present():

    with pytest.raises(ValueError, match=r"Either unsupported_models or supported_models must be provided"):

        class BadNone(
                BaseModel,
                ModelGatedFieldMixin[int],
                field_name="dummy",
                default_if_supported=1,
        ):
            dummy: int | None = Field(default=None)
            model_name: str = "alpha"

        _ = BadNone()


def test_model_gated_field_default_applied_when_supported_and_value_none():

    class GoodSupported(
            BaseModel,
            ModelGatedFieldMixin[int],
            field_name="dummy",
            default_if_supported=5,
            supported_models=(re.compile(r"^alpha$"), ),
    ):
        dummy: int | None = Field(default=None)
        model_name: str

    m = GoodSupported(model_name="alpha")
    assert m.dummy == 5


def test_model_gated_field_raises_when_not_supported_and_value_set():

    class GoodUnsupported(
            BaseModel,
            ModelGatedFieldMixin[int],
            field_name="dummy",
            default_if_supported=5,
            unsupported_models=(re.compile(r"alpha"), ),
    ):
        dummy: int | None = Field(default=None)
        model_name: str

    with pytest.raises(ValidationError, match=r"dummy is not supported for model_name: alpha"):
        _ = GoodUnsupported(model_name="alpha", dummy=3)


def test_model_gated_field_none_returned_when_not_supported_and_value_none():

    class GoodUnsupported(
            BaseModel,
            ModelGatedFieldMixin[int],
            field_name="dummy",
            default_if_supported=5,
            unsupported_models=(re.compile(r"alpha"), ),
    ):
        dummy: int | None = Field(default=None)
        model_name: str

    m = GoodUnsupported(model_name="alpha")
    assert m.dummy is None


def test_model_gated_field_default_applied_when_no_model_key_present():

    class NoModelKey(
            BaseModel,
            ModelGatedFieldMixin[int],
            field_name="dummy",
            default_if_supported=7,
            supported_models=(re.compile(r"anything"), ),
    ):
        dummy: int | None = Field(default=None)

    # No model_keys are present in the data, so value falls back to default_if_supported
    m = NoModelKey()
    assert m.dummy == 7


def test_model_gated_field_with_custom_model_keys():

    class CustomKeys(
            BaseModel,
            ModelGatedFieldMixin[int],
            field_name="dummy",
            default_if_supported=9,
            unsupported_models=(re.compile(r"ban"), ),
            model_keys=("custom_key", ),
    ):
        dummy: int | None = Field(default=None)
        custom_key: str

    # Unsupported because custom_key matches the banned pattern
    m = CustomKeys(custom_key="banned")
    assert m.dummy is None


def test_model_gated_field_with_custom_model_keys_and_default_if_supported():

    class CustomKeys(
            BaseModel,
            ModelGatedFieldMixin[int],
            field_name="dummy",
            default_if_supported=9,
            unsupported_models=(re.compile(r"ban"), ),
            model_keys=("custom_key", ),
    ):
        dummy: int | None = Field(default=None)
        custom_key: str

    # Supported because custom_key does not match the banned pattern
    m = CustomKeys(custom_key="valid")
    assert m.dummy == 9
