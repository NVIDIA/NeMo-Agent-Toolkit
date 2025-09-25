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
from pathlib import Path


def locate_example_dir(example_config_class: type) -> importlib.resources.abc.Traversable:
    """
    Locate the example directory for an example's config class.
    """
    package_name = inspect.getmodule(example_config_class).__package__
    return importlib.resources.files(package_name)


def locate_example_config(example_config_class: type, config_file: str = "config.yml") -> Path:
    """
    Locate the example config file for an example's config class, assumes the example contains a 'configs' directory
    directly under the example directory, or a symlink to it.
    """
    example_dir = locate_example_dir(example_config_class)
    return example_dir.joinpath("configs", config_file).absolute()
