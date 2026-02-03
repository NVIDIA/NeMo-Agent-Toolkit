# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import pytest

from nat.runtime.loader import load_config
from nat.utils import run_workflow
from nat.utils.type_converter import GlobalTypeConverter


@pytest.mark.usefixtures("reset_global_type_converter")
@pytest.mark.parametrize("to_type", [str, int, float])
@pytest.mark.parametrize("use_pathlib", [True, False])
@pytest.mark.parametrize("use_config_object", [True, False])
async def test_run_workflow(echo_config_file: str, use_pathlib: bool, use_config_object: bool, to_type: type) -> None:
    if use_pathlib:
        config_file = Path(echo_config_file)
    else:
        config_file = echo_config_file

    config = None
    if use_config_object:
        config = load_config(config_file)
        config_file = None

    if to_type is not str:

        def converter(x: str) -> to_type:
            return to_type(x)

        GlobalTypeConverter.register_converter(converter)

    prompt = "55"
    expected_result = to_type(prompt)
    result = await run_workflow(config_file=config_file, config=config, prompt=prompt, to_type=to_type)
    assert isinstance(result, to_type)
    assert result == expected_result
