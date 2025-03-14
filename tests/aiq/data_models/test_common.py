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

import json
import typing
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from aiq.data_models import common


class ԊashableTĕstModel(common.HashableBaseModel):  # pylint: disable=non-ascii-name
    """
    Intentionally using non-ascci characters to test the encoding for the hash
    """
    apples: int
    pair: tuple[int, int]


def test_hashable_base_model_is_hashable():
    h1 = ԊashableTĕstModel(apples=2, pair=(4, 5))
    h2 = ԊashableTĕstModel(apples=3, pair=(4, 5))
    h3 = ԊashableTĕstModel(apples=2, pair=(4, 5))  # same as h1

    configs = {h1, h2, h3}
    assert len(configs) == 2
    assert h1 in configs
    assert h2 in configs
    assert h3 in configs


def test_hashable_base_model_write_json_schema(tmp_path: Path):
    schema_path = tmp_path / "test_schema.json"
    ԊashableTĕstModel.write_json_schema(schema_path)

    assert schema_path.exists()
    assert schema_path.is_file()

    with open(schema_path, "r", encoding="utf-8") as f:
        schema = json.load(f)
        assert schema == ԊashableTĕstModel.generate_json_schema()


def test_subclass_depth():

    class Parent:
        pass

    class Child(Parent):
        pass

    class GrandChild(Child):
        pass

    assert common.subclass_depth(GrandChild) == 3

    # We know that ԊashableTĕstModel has at least three levels of inheritance:
    # ԊashableTĕstModel -> HashableBaseModel -> BaseModel -> ... -> object
    # we don't want to make any assumptions about the number of levels of inheritance between BaseModel and object
    assert common.subclass_depth(ԊashableTĕstModel) >= 3


@pytest.mark.parametrize("v, expected_value",
                         [({
                             "_type": "_type_test"
                         }, "_type_test"), ({
                             "type": "type_test"
                         }, "type_test"), ({
                             "_type": "correct", "type": "incorrect"
                         }, "correct"), ({}, None), (MagicMock(spec=["type"], type="apples"), "apples")],
                         ids=["dict-with-_type", "dict-with-type", "dict with both", "no_type", "object"])
def test_type_discriminator(v: typing.Any, expected_value: str | None):
    assert common.TypedBaseModel.discriminator(v) == expected_value
