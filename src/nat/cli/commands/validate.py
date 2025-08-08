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

from pathlib import Path

import click


@click.command()
@click.option("--config_file",
              type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
              required=True,
              help="Configuration file to validate")
def validate_command(config_file: Path):
    """Validate a configuration file"""
    # load function level dependencies
    from io import StringIO

    from nat.runtime.loader import load_config

    try:
        click.echo(f"Validating configuration file: {config_file}")
        config = load_config(config_file)
        click.echo(click.style("✓ Configuration file is valid!", fg="green"))

        stream = StringIO()

        config.print_summary(stream=stream)

        click.echo_via_pager(stream.getvalue())
    except Exception as e:
        click.echo(click.style("✗ Validation failed!\n\nError:", fg="red"))

        click.echo(click.style(e, fg="red"))
        raise click.ClickException(str(e)) from e
