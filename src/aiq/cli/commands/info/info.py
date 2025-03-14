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

import logging

import click

from aiq.cli.commands.info.list_channels import list_channels
from aiq.cli.commands.info.list_components import list_components

logger = logging.getLogger(__name__)


@click.group(name=__name__,
             invoke_without_command=False,
             help="Provide information about the local AgentIQ environment.")
def info_command(**kwargs):
    """
    Provide information about the local AgentIQ environment.
    """
    pass


info_command.add_command(list_components, name="components")
info_command.add_command(list_channels, "channels")
