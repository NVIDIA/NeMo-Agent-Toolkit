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
"""
Comprehensive tests for environment variable handling and API key fixtures.
"""

import os

import pytest

from nat.test.plugin import require_env_variables


@pytest.mark.usefixtures("restore_environ")
@pytest.mark.parametrize("fail_on_missing", [True, False])
@pytest.mark.parametrize("env_vars",
                         [{
                             "SOME_KEY": "xyz"
                         }, {
                             "SOME_KEY": "xyz", "OTHER_KEY": "abc"
                         }, {
                             "SOME_KEY": "xyz", "OTHER_KEY": "abc", "MISSING_KEY": None
                         }, {
                             "SOME_KEY": "xyz", "OTHER_KEY": "abc", "MISSING_KEY": None, "EMPTY_KEY": None
                         }])
def test_require_env_variables(fail_on_missing: bool, env_vars: dict[str, str | None]):
    # Note the variable name `fail_on_missing` is used to avoid conflict with the `fail_missing` fixture
    has_missing = False
    var_names = []
    for (env_var, value) in env_vars.items():
        var_names.append(env_var)
        if value is not None:
            os.environ[env_var] = value
        else:
            has_missing = True
            os.environ.pop(env_var, None)

    if has_missing:
        if fail_on_missing:
            expected_exception = RuntimeError
        else:
            expected_exception = pytest.skip.Exception

        with pytest.raises(expected_exception, match="unittest"):
            require_env_variables(varnames=var_names, reason="unittest", fail_missing=fail_on_missing)

    else:
        assert require_env_variables(varnames=var_names, reason="unittest", fail_missing=fail_on_missing) == env_vars
