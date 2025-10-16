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

from pathlib import Path

import pytest

from nat.test import utils


@pytest.fixture(name="example_config_class")
def example_config_class_fixture() -> type:
    from nat_simple_web_query.register import WebQueryToolConfig
    return WebQueryToolConfig


@pytest.fixture(name="simple_web_query_dir")
def simple_web_query_dir_fixture(root_repo_dir: Path) -> Path:
    # This fixture will need to be updated if the example is moved or removed
    return root_repo_dir.joinpath("examples", "getting_started", "simple_web_query")


@pytest.fixture(name="simple_web_query_src_dir")
def simple_web_query_src_dir_fixture(simple_web_query_dir: Path) -> Path:
    return simple_web_query_dir.joinpath("src", "nat_simple_web_query")


def test_locate_example_src_dir(example_config_class: type, simple_web_query_src_dir: Path):
    example_dir = utils.locate_example_src_dir(example_config_class)
    assert example_dir == simple_web_query_src_dir


def test_locate_example_dir(example_config_class: type, simple_web_query_dir: Path):
    example_dir = utils.locate_example_dir(example_config_class)
    assert example_dir == simple_web_query_dir


@pytest.mark.parametrize("config_file_name, exists", [("config.yml", True), ("nonexistent.yml", False)])
@pytest.mark.parametrize("assert_exists", [True, False])
def test_locate_example_config(example_config_class: type,
                               simple_web_query_src_dir: Path,
                               config_file_name: str,
                               exists: bool,
                               assert_exists: bool):
    expected_config_path = simple_web_query_src_dir.joinpath("configs", config_file_name).absolute()
    if not exists and assert_exists:
        with pytest.raises(AssertionError, match="does not exist"):
            utils.locate_example_config(example_config_class, config_file_name, assert_exists)

    else:
        config_path = utils.locate_example_config(example_config_class, config_file_name, assert_exists)
        assert config_path == expected_config_path
        assert (exists == config_path.exists())
