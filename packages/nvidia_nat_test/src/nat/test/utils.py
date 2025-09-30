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

import importlib.resources
import inspect
import subprocess
import typing
from pathlib import Path

if typing.TYPE_CHECKING:
    from nat.data_models.config import Config
    from nat.utils.type_utils import StrPath


def locate_repo_root() -> Path:
    result = subprocess.run(["git", "rev-parse", "--show-toplevel"], check=False, capture_output=True, text=True)
    assert result.returncode == 0, f"Failed to get git root: {result.stderr}"
    return Path(result.stdout.strip())


def locate_example_src_dir(example_config_class: type) -> Path:
    """
    Locate the example src directory for an example's config class.
    """
    package_name = inspect.getmodule(example_config_class).__package__
    return importlib.resources.files(package_name)


def locate_example_dir(example_config_class: type) -> Path:
    """
    Locate the example directory for an example's config class.
    """
    src_dir = locate_example_src_dir(example_config_class)
    example_dir = src_dir.parent.parent
    return example_dir


def locate_example_config(example_config_class: type,
                          config_file: str = "config.yml",
                          assert_exists: bool = True) -> Path:
    """
    Locate the example config file for an example's config class, assumes the example contains a 'configs' directory
    """
    example_dir = locate_example_src_dir(example_config_class)
    config_path = example_dir.joinpath("configs", config_file).absolute()
    if assert_exists:
        assert config_path.exists(), f"Config file {config_path} does not exist"

    return config_path


async def run_workflow(
    config_file: "StrPath | None",
    question: str,
    expected_answer: str,
    assert_expected_answer: bool = True,
    config: "Config | None" = None,
) -> str:
    from nat.builder.workflow_builder import WorkflowBuilder
    from nat.runtime.loader import load_config
    from nat.runtime.session import SessionManager

    if config is None:
        assert config_file is not None, "Either config_file or config must be provided"
        config = load_config(config_file)

    async with WorkflowBuilder.from_config(config=config) as workflow_builder:
        workflow = SessionManager(await workflow_builder.build())
        async with workflow.run(question) as runner:
            result = await runner.result(to_type=str)

    if assert_expected_answer:
        assert expected_answer.lower() in result.lower(), f"Expected '{expected_answer}' in '{result}'"

    return result
