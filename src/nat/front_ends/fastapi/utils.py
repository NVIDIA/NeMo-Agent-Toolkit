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

import importlib
import os


def get_config_file_path() -> str:
    """
    Get the path to the NAT configuration file from the environment variable NAT_CONFIG_FILE.
    Raises ValueError if the environment variable is not set.
    """
    config_file_path = os.getenv("NAT_CONFIG_FILE")
    if (not config_file_path):
        raise ValueError("Config file not found in environment variable NAT_CONFIG_FILE.")

    return os.path.abspath(config_file_path)


def import_class_from_string(class_full_name: str) -> type:
    """
    Import a class from a string in the format 'module.submodule.ClassName'.
    Raises ImportError if the class cannot be imported.
    """
    try:
        class_name_parts = class_full_name.split(".")

        module_name = ".".join(class_name_parts[:-1])
        class_name = class_name_parts[-1]

        module = importlib.import_module(module_name)

        if not hasattr(module, class_name):
            raise ValueError(f"Class '{class_full_name}' not found.")

        return getattr(module, class_name)
    except (ImportError, AttributeError) as e:
        raise ImportError(f"Could not import {class_full_name}.") from e


def get_class_name(cls: type) -> str:
    """
    Get the full class name including the module.
    """
    return f"{cls.__module__}.{cls.__qualname__}"
