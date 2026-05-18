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

from pathlib import Path

from nat import plugin_api
from nat.builder.builder import Builder
from nat.builder.function import FunctionGroup
from nat.cli.register_workflow import register_function
from nat.cli.register_workflow import register_function_group
from nat.data_models.function import FunctionBaseConfig
from nat.data_models.function import FunctionGroupBaseConfig


def test_plugin_api_exports_core_function_authoring_surface():
    assert plugin_api.Builder is Builder
    assert plugin_api.FunctionBaseConfig is FunctionBaseConfig
    assert plugin_api.FunctionGroupBaseConfig is FunctionGroupBaseConfig
    assert plugin_api.FunctionGroup is FunctionGroup
    assert plugin_api.register_function is register_function
    assert plugin_api.register_function_group is register_function_group


def test_plugin_api_all_exports_are_importable():
    for name in plugin_api.__all__:
        assert getattr(plugin_api, name) is not None


def test_plugin_authoring_docs_prefer_public_api_imports():
    repo_root = Path(__file__).parents[4]
    paths = [
        repo_root / "docs/source/components/sharing-components.md",
        repo_root / "docs/source/build-workflows/advanced/middleware.md",
        repo_root / "docs/source/build-workflows/functions-and-function-groups/function-groups.md",
        repo_root / "docs/source/extend/custom-components",
        repo_root / "docs/source/extend/plugins.md",
        repo_root / "docs/source/improve-workflows/evaluate.md",
        repo_root / "docs/source/improve-workflows/optimizer.md",
        repo_root / "docs/source/improve-workflows/test-time-compute.md",
        repo_root / "packages/nvidia_nat_core/src/nat/cli/commands/workflow/templates/workflow.py.j2",
    ]
    denied_patterns = [
        "nat.cli.register_workflow",
        "nat.cli.register_llm_client",
        "from nat.builder.builder import Builder",
        "from nat.builder.builder import EvalBuilder",
        "from nat.builder.function import FunctionGroup",
        "from nat.builder.function_info import FunctionInfo",
        "from nat.data_models.component_ref import",
        "from nat.data_models.function import FunctionBaseConfig",
        "from nat.data_models.function import FunctionGroupBaseConfig",
    ]

    files: list[Path] = []
    for path in paths:
        if path.is_dir():
            files.extend(path.rglob("*.md"))
        else:
            files.append(path)

    violations = []
    for file_path in files:
        text = file_path.read_text(encoding="utf-8")
        for pattern in denied_patterns:
            if pattern in text:
                violations.append(f"{file_path.relative_to(repo_root)} contains {pattern!r}")

    assert not violations, "Plugin authoring docs should use nat.plugin_api:\n" + "\n".join(violations)
