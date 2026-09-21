# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from pydantic import BaseModel
from pydantic import model_validator

from nat.utils.data_models.schema_validator import validate_schema


class _Inner(BaseModel):
    port: int


class _Config(BaseModel):
    name: str
    inner: _Inner
    items: list[_Inner] = []

    @model_validator(mode="after")
    def _reject_bad_name(self) -> "_Config":
        if self.name == "bad":
            raise ValueError("name may not be 'bad'")
        return self


def test_validate_schema_reports_full_nested_location():
    with pytest.raises(ValueError, match=r"^Invalid configuration: inner\.port: Input should be a valid integer"):
        validate_schema({"name": "ok", "inner": {"port": "x"}}, _Config)


def test_validate_schema_includes_list_indices_in_location():
    with pytest.raises(ValueError, match=r"items\.1\.port: Input should be a valid integer"):
        validate_schema({"name": "ok", "inner": {"port": 1}, "items": [{"port": 1}, {"port": "x"}]}, _Config)


def test_validate_schema_joins_multiple_errors():
    with pytest.raises(ValueError, match=r"name: Field required; inner\.port: Input should be a valid integer"):
        validate_schema({"inner": {"port": "x"}}, _Config)


def test_validate_schema_reports_model_level_errors_without_location():
    # A model validator error has an empty ``loc``; it used to escape as an IndexError.
    with pytest.raises(ValueError, match=r"^Invalid configuration: Value error, name may not be 'bad'$"):
        validate_schema({"name": "bad", "inner": {"port": 1}}, _Config)


def test_validate_schema_returns_model_on_success():
    config = validate_schema({"name": "ok", "inner": {"port": 1}}, _Config)
    assert isinstance(config, _Config)
    assert config.inner.port == 1
