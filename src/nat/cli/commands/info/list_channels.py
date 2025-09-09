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

logger = logging.getLogger(__name__)


@click.group(name=__name__, invoke_without_command=True, help="List the configured remote registry channels.")
@click.option("-t", "--type", "channel_type", type=str, required=False, help=("Filter the results by channel type."))
def list_channels(channel_type: str):
    from nat.settings.global_settings import GlobalSettings

    settings = GlobalSettings().get()
    try:
        settings.print_channel_settings(channel_type=channel_type)
    except Exception as e:
        logger.exception("Error listing channels: %s", e)
