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
"""Unit tests for example Harbor adapter helpers."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


def _load_adapter_common_module():
    repo_root = Path(__file__).resolve().parents[3]
    module_path = repo_root / "examples/evaluation_and_profiling/simple_calculator_eval/harbor_adapters/common.py"
    spec = importlib.util.spec_from_file_location("harbor_adapter_common_for_test", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_copy_template_ignores_cache_artifacts(tmp_path, monkeypatch) -> None:
    common = _load_adapter_common_module()
    template_dir = tmp_path / "template"
    tests_dir = template_dir / "tests"
    nested_pycache_dir = tests_dir / "__pycache__"
    nested_pycache_dir.mkdir(parents=True)
    (tests_dir / "test.sh").write_text("echo ok\n", encoding="utf-8")
    (nested_pycache_dir / "evaluate.cpython-313.pyc").write_bytes(b"\0\0")
    (template_dir / ".pytest_cache").mkdir()
    (template_dir / "top_level.pyc").write_bytes(b"\0\0")
    monkeypatch.setattr(common, "TEMPLATE_DIR", template_dir)

    output_dir = tmp_path / "output"
    output_dir.mkdir()
    common.copy_template(output_dir)

    assert (output_dir / "tests" / "test.sh").exists()
    assert not (output_dir / "tests" / "__pycache__").exists()
    assert not (output_dir / ".pytest_cache").exists()
    assert not (output_dir / "top_level.pyc").exists()


def test_parse_basic_arithmetic_rejects_division_by_zero() -> None:
    common = _load_adapter_common_module()

    with pytest.raises(ValueError, match="Division by zero"):
        common.parse_basic_arithmetic("What is 4 divided by 0?")


def test_resolve_safe_task_dir_rejects_path_traversal(tmp_path: Path) -> None:
    common = _load_adapter_common_module()

    assert common.resolve_safe_task_dir(tmp_path, "task-1") == tmp_path.resolve() / "task-1"
    for invalid_task_id in ("../escape", "/tmp/escape", "nested/../../escape"):
        with pytest.raises(ValueError, match="Invalid local_task_id"):
            common.resolve_safe_task_dir(tmp_path, invalid_task_id)
