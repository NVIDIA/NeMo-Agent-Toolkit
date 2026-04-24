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

"""Tests for generic nat-harbor adapter CLI utilities."""

from __future__ import annotations

import pytest

from nat_harbor.adapters import BaseTaskAdapter
from nat_harbor.cli import load_adapter_class
from nat_harbor.cli import parse_source_ids


class _TestAdapter(BaseTaskAdapter):
    adapter_name = "test"

    def list_available_tasks(self) -> list[str]:
        return ["1"]

    def generate_many(self, source_ids, output_dir, overwrite=False):
        del source_ids, output_dir, overwrite
        return 0, 0


def test_parse_source_ids_none() -> None:
    assert parse_source_ids(None) is None


def test_parse_source_ids_csv() -> None:
    assert parse_source_ids("1, 2 ,3") == ["1", "2", "3"]


def test_load_adapter_class_invalid_format() -> None:
    with pytest.raises(ValueError, match="Invalid --adapter-class value"):
        load_adapter_class("bad-format")


def test_load_adapter_class_wrong_type() -> None:
    with pytest.raises(ValueError, match="BaseTaskAdapter subclass"):
        load_adapter_class("nat_harbor.cli:build_parser")


def test_load_adapter_class_success(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("nat_harbor.cli.importlib.import_module", lambda _: __import__(__name__))
    cls = load_adapter_class(f"{__name__}:_TestAdapter")
    assert issubclass(cls, BaseTaskAdapter)

