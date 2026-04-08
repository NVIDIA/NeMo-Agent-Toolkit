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

from nat.plugins.eval.cli import evaluate as cli_evaluate
from nat.plugins.eval.runtime import evaluate as runtime_evaluate


def test_runtime_full_dependency_error_includes_install_hint():
    with pytest.raises(ModuleNotFoundError, match=r'nvidia-nat-eval\[full\]'):
        runtime_evaluate._raise_full_eval_dependency_error(ImportError("mock missing dependency"))


def test_cli_full_dependency_error_includes_install_hint():
    with pytest.raises(ModuleNotFoundError, match=r'nvidia-nat-eval\[full\]'):
        cli_evaluate._raise_full_eval_dependency_error(ImportError("mock missing dependency"))
