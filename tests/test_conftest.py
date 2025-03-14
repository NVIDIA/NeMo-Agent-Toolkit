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

import pytest

from aiq.cli.register_workflow import register_function
from aiq.cli.type_registry import GlobalTypeRegistry
from aiq.cli.type_registry import TypeRegistry
from aiq.data_models.function import FunctionBaseConfig


@pytest.fixture(name="registry_counter", scope="module")
def registry_counter_fixture():

    return {"functions": len(GlobalTypeRegistry.get()._registered_functions)}


@pytest.mark.parametrize("test_iter", [0, 1])
def test_registry_fixture(registry: TypeRegistry, test_iter: int, registry_counter: dict[str, int]):

    assert len(registry._registered_functions) == registry_counter["functions"]

    if test_iter == 0:
        # Add some entries, if the fixture is working properly the entries should be reset, and won't be present in the
        # next test iteration
        class TestFunctionConfig(FunctionBaseConfig, name="test_function"):
            pass

        @register_function(config_type=TestFunctionConfig)
        async def test_function(config: TestFunctionConfig, builder):
            yield lambda: None

        assert len(registry._registered_functions) == registry_counter["functions"] + 1
