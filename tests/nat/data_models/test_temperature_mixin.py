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

import pytest
from pydantic import ValidationError

from nat.data_models.temperature_mixin import TemperatureMixin


def test_temperature_default_when_supported_and_none():

    class TestConfig(TemperatureMixin):  # type: ignore
        model_name: str

    m = TestConfig(model_name="gpt-4o")
    assert m.temperature == 0.0


def test_temperature_respects_value_when_supported():

    class TestConfig(TemperatureMixin):  # type: ignore
        model_name: str

    m = TestConfig(model_name="gpt-4o", temperature=0.3)
    assert m.temperature == 0.3


def test_temperature_rejected_when_unsupported_and_set():

    class TestConfig(TemperatureMixin):  # type: ignore
        model_name: str

    with pytest.raises(ValidationError, match=r"temperature is not supported for model_name: gpt5"):
        _ = TestConfig(model_name="gpt5", temperature=0.2)

    with pytest.raises(ValidationError, match=r"temperature is not supported for model_name: gpt5o"):
        _ = TestConfig(model_name="gpt5o", temperature=0.2)

    with pytest.raises(ValidationError, match=r"temperature is not supported for model_name: gpt-5"):
        _ = TestConfig(model_name="gpt-5", temperature=0.2)

    with pytest.raises(ValidationError, match=r"temperature is not supported for model_name: gpt-5o"):
        _ = TestConfig(model_name="gpt-5o", temperature=0.2)


def test_temperature_none_when_unsupported_and_value_none():

    class TestConfig(TemperatureMixin):  # type: ignore
        model_name: str

    m = TestConfig(model_name="gpt5")
    assert m.temperature is None


def test_temperature_range_validation():

    class TestConfig(TemperatureMixin):  # type: ignore
        model_name: str

    with pytest.raises(ValidationError):
        _ = TestConfig(model_name="gpt-4o", temperature=-0.1)

    with pytest.raises(ValidationError):
        _ = TestConfig(model_name="gpt-4o", temperature=1.1)


def test_temperature_default_when_no_model_keys_present():
    # No model keys present; falls back to default_if_supported
    m = TemperatureMixin()
    assert m.temperature == 0.0
