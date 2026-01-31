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

import click


def test_a2a_plugin_discovered():
    """Test that A2A plugin is discovered when nvidia-nat-a2a is installed."""
    import nat.plugins.a2a.cli.commands  # noqa: F401
    from nat.cli.plugin_loader import discover_and_load_cli_plugins

    cli_group = click.Group()
    discover_and_load_cli_plugins(cli_group)

    # A2A should be discovered and loaded
    assert "a2a" in cli_group.commands
