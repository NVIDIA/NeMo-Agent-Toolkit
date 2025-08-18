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

from nat.data_models.supported_field import SupportedField


def test_supported_field_requires_one_selector():

    class BadBoth(
            BaseModel,
            SupportedField[int],
            field_name="dummy",
            default_if_supported=1,
            unsupported_models=(re.compile(r"alpha"), ),
            supported_models=(re.compile(r"beta"), ),
    ):
        dummy: int | None = Field(default=None)
        model_name: str = "alpha"

    with pytest.raises(ValidationError, match=r"Only one of unsupported_models or supported_models must be provided"):
        _ = BadBoth()


def test_supported_field_requires_selector_present():

    class BadNone(
            BaseModel,
            SupportedField[int],
            field_name="dummy",
            default_if_supported=1,
    ):
        dummy: int | None = Field(default=None)
        model_name: str = "alpha"

    with pytest.raises(ValidationError, match=r"Either unsupported_models or supported_models must be provided"):
        _ = BadNone()


def test_supported_field_default_applied_when_supported_and_value_none():

    class GoodSupported(
            BaseModel,
            SupportedField[int],
            field_name="dummy",
            default_if_supported=5,
            supported_models=(re.compile(r"^alpha$"), ),
    ):
        dummy: int | None = Field(default=None)
        model_name: str

    m = GoodSupported(model_name="alpha")
    assert m.dummy == 5


def test_supported_field_raises_when_not_supported_and_value_set():

    class GoodUnsupported(
            BaseModel,
            SupportedField[int],
            field_name="dummy",
            default_if_supported=5,
            unsupported_models=(re.compile(r"alpha"), ),
    ):
        dummy: int | None = Field(default=None)
        model_name: str

    with pytest.raises(ValidationError, match=r"dummy is not supported for model_name: alpha"):
        _ = GoodUnsupported(model_name="alpha", dummy=3)


def test_supported_field_none_returned_when_not_supported_and_value_none():

    class GoodUnsupported(
            BaseModel,
            SupportedField[int],
            field_name="dummy",
            default_if_supported=5,
            unsupported_models=(re.compile(r"alpha"), ),
    ):
        dummy: int | None = Field(default=None)
        model_name: str

    m = GoodUnsupported(model_name="alpha")
    assert m.dummy is None


def test_supported_field_default_applied_when_no_model_key_present():

    class NoModelKey(
            BaseModel,
            SupportedField[int],
            field_name="dummy",
            default_if_supported=7,
            supported_models=(re.compile(r"anything"), ),
    ):
        dummy: int | None = Field(default=None)

    # No model_keys are present in the data, so value falls back to default_if_supported
    m = NoModelKey()
    assert m.dummy == 7


def test_supported_field_with_custom_model_keys():

    class CustomKeys(
            BaseModel,
            SupportedField[int],
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
