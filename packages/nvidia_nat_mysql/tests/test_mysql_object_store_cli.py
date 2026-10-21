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

from nat.cli.commands.object_store.object_store import object_store_command
from nat.plugins.mysql.object_store import MySQLObjectStoreClientConfig


def test_mysql_cli_parameters_map_to_config_fields():
    """Every `nat object-store mysql` parameter must set a config field; unknown fields are silently dropped."""
    mysql_group = object_store_command.commands["mysql"]

    parameter_names = {param.name for param in mysql_group.params}

    assert parameter_names - set(MySQLObjectStoreClientConfig.model_fields) == set()
