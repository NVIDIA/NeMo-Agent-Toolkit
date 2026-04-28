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


class _BlockAioboto3:

    def find_spec(self, fullname, path=None, target=None):  # noqa: ANN001
        if fullname == "aioboto3" or fullname.startswith("aioboto3."):
            raise ModuleNotFoundError("No module named 'aioboto3'")
        return None


def test_runtime_full_dependency_error_includes_install_hint():
    from nat.plugins.eval.runtime import evaluate as runtime_evaluate

    with pytest.raises(ModuleNotFoundError, match=r'nvidia-nat-eval\[full\]'):
        runtime_evaluate._raise_full_eval_dependency_error(ImportError("mock missing dependency"))


def test_cli_full_dependency_error_includes_install_hint():
    from nat.plugins.eval.cli import evaluate as cli_evaluate

    with pytest.raises(ModuleNotFoundError, match=r'nvidia-nat-eval\[full\]'):
        cli_evaluate._raise_full_eval_dependency_error(ImportError("mock missing dependency"))


def test_runtime_evaluate_import_does_not_require_aioboto3(monkeypatch):
    import importlib
    import sys

    module_names = (
        "aioboto3",
        "nat.plugins.eval.runtime.evaluate",
        "nat.plugins.eval.utils.output_uploader",
    )
    original_modules = {name: sys.modules.get(name) for name in module_names}
    runtime_pkg = sys.modules.get("nat.plugins.eval.runtime")
    utils_pkg = sys.modules.get("nat.plugins.eval.utils")
    had_runtime_evaluate = hasattr(runtime_pkg, "evaluate") if runtime_pkg is not None else False
    had_utils_output_uploader = hasattr(utils_pkg, "output_uploader") if utils_pkg is not None else False
    original_runtime_evaluate = getattr(runtime_pkg, "evaluate", None) if runtime_pkg is not None else None
    original_utils_output_uploader = getattr(utils_pkg, "output_uploader", None) if utils_pkg is not None else None
    for name in module_names:
        sys.modules.pop(name, None)

    monkeypatch.setattr(sys, "meta_path", [_BlockAioboto3(), *sys.meta_path])
    try:
        runtime_evaluate = importlib.import_module("nat.plugins.eval.runtime.evaluate")
        assert runtime_evaluate.EvaluationRun.__name__ == "EvaluationRun"
    finally:
        for name in module_names:
            sys.modules.pop(name, None)
        for name, module in original_modules.items():
            if module is not None:
                sys.modules[name] = module
        if runtime_pkg is not None:
            if had_runtime_evaluate:
                runtime_pkg.evaluate = original_runtime_evaluate
            elif hasattr(runtime_pkg, "evaluate"):
                del runtime_pkg.evaluate
        if utils_pkg is not None:
            if had_utils_output_uploader:
                utils_pkg.output_uploader = original_utils_output_uploader
            elif hasattr(utils_pkg, "output_uploader"):
                del utils_pkg.output_uploader
