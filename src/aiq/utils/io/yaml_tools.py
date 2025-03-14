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

import io
import typing

import yaml

from aiq.utils.type_utils import StrPath


def yaml_load(config_path: StrPath) -> dict:

    # Read YAML file
    with open(config_path, 'r', encoding="utf-8") as stream:
        config_data = yaml.safe_load(stream)

    return config_data


def yaml_loads(config: str) -> dict:

    stream = io.StringIO(config)
    stream.seek(0)

    return yaml.safe_load(stream)


def yaml_dump(config: dict, fp: typing.TextIO) -> None:

    yaml.dump(config, stream=fp, indent=2, sort_keys=False)

    fp.flush()


def yaml_dumps(config: dict) -> str:

    return yaml.dump(config, indent=2)
