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

# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import os
import subprocess
import sys
import re
import shutil
from pathlib import Path

import click
from jinja2 import Environment, FileSystemLoader, select_autoescape

from nat.utils.repo import get_repo_root

logger = logging.getLogger(__name__)


@click.command("create")
@click.argument("workflow_name")
@click.option(
    "--install",
    is_flag=True,
    help="Install the workflow after creation.",
)
@click.option(
    "--workflow-dir",
    default=".",
    help="Directory where the workflow should be created.",
)
@click.option(
    "--description",
    default="A new workflow",
    help="Description of the workflow.",
)
def create_command(workflow_name: str, install: bool, workflow_dir: str, description: str) -> None:
    """Create a new workflow project from templates.

    Args:
        workflow_name (str): Human-friendly workflow name. Hyphens allowed; will be converted to a Python-safe package.
        install (bool): If True, installs the workflow package after generation.
        workflow_dir (str): Output directory where the workflow folder is created.
        description (str): Text used in generated module docstrings and metadata.
    """
    # Validate workflow name
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_-]*", workflow_name):
        raise click.BadParameter(  # noqa: TRY003
            "Workflow name must start with a letter/underscore and contain only letters, digits, hyphens, or underscores."
        )

    # Resolve paths
    new_workflow_dir = Path(workflow_dir) / workflow_name
    if new_workflow_dir.exists():
        raise click.ClickException(f"Workflow '{workflow_name}' already exists.")

    repo_root = get_repo_root()
    try:
        rel_path_to_repo_root = "" if not repo_root else os.path.relpath(repo_root, new_workflow_dir)
    except ValueError:
        rel_path_to_repo_root = ""

    try:
        # Setup Jinja2 environment
        template_dir = Path(__file__).parent / "templates" / "workflow"
        env = Environment(
            loader=FileSystemLoader(str(template_dir)),
            autoescape=select_autoescape(enabled_extensions=("html", "xml"), default_for_string=False),
        )

        context = {
            "workflow_name": workflow_name,
            "description": description,
            "rel_path_to_repo_root": rel_path_to_repo_root,
        }

        new_workflow_dir.mkdir(parents=True)

        for template_file in template_dir.rglob("*"):
            if template_file.is_file():
                relative_path = template_file.relative_to(template_dir)
                target_file = new_workflow_dir / relative_path
                target_file.parent.mkdir(parents=True, exist_ok=True)

                template = env.get_template(str(relative_path))
                target_file.write_text(template.render(context))

        click.echo(f"Workflow '{workflow_name}' created at {new_workflow_dir}")

        if install:
            install_cmd = (
                ["uv", "pip", "install", "-e", str(new_workflow_dir)]
                if repo_root
                else [sys.executable, "-m", "pip", "install", "-e", str(new_workflow_dir)]
            )
            try:
                subprocess.run(install_cmd, capture_output=True, text=True, check=True)  # noqa: S603
            except subprocess.CalledProcessError as e:
                logger.error("Installation failed (exit %s): %s", e.returncode, e.stderr)
                raise click.ClickException("Installation failed. See logs for details.") from e

    except Exception as e:
        logger.exception("Error while creating workflow")
        raise click.ClickException("Unexpected error while creating workflow. See logs for details.") from e


@click.command("reinstall")
@click.argument("workflow_name")
def reinstall_command(workflow_name: str) -> None:
    """Reinstall an existing workflow package in editable mode."""
    try:
        repo_root = get_repo_root()
        workflow_dir = Path(repo_root) / workflow_name if repo_root else Path(workflow_name)

        if not workflow_dir.exists():
            raise click.ClickException(f"Workflow '{workflow_name}' does not exist.")

        reinstall_cmd = (
            ["uv", "pip", "install", "-e", str(workflow_dir)]
            if repo_root
            else [sys.executable, "-m", "pip", "install", "-e", str(workflow_dir)]
        )
        try:
            subprocess.run(reinstall_cmd, capture_output=True, text=True, check=True)  # noqa: S603
        except subprocess.CalledProcessError as e:
            logger.error("Reinstallation failed (exit %s): %s", e.returncode, e.stderr)
            raise click.ClickException("Reinstallation failed. See logs for details.") from e

        click.echo(f"Workflow '{workflow_name}' reinstalled successfully.")

    except Exception as e:
        logger.exception("Error while reinstalling workflow")
        raise click.ClickException("Unexpected error while reinstalling workflow. See logs for details.") from e


@click.command("delete")
@click.argument("workflow_name")
def delete_command(workflow_name: str) -> None:
    """Delete a workflow package and uninstall it."""
    try:
        repo_root = get_repo_root()
        workflow_dir = Path(repo_root) / workflow_name if repo_root else Path(workflow_name)

        if not workflow_dir.exists():
            raise click.ClickException(f"Workflow '{workflow_name}' does not exist.")

        uninstall_cmd = (
            ["uv", "pip", "uninstall", "-y", workflow_name]
            if repo_root
            else [sys.executable, "-m", "pip", "uninstall", "-y", workflow_name]
        )
        try:
            subprocess.run(uninstall_cmd, capture_output=True, text=True, check=True)  # noqa: S603
        except subprocess.CalledProcessError as e:
            logger.error("Uninstallation failed (exit %s): %s", e.returncode, e.stderr)
            raise click.ClickException("Uninstallation failed. See logs for details.") from e

        # Remove workflow directory cleanly
        shutil.rmtree(workflow_dir)

        click.echo(f"Workflow '{workflow_name}' deleted successfully.")

    except Exception as e:
        logger.exception("Error while deleting workflow")
        raise click.ClickException("Unexpected error while deleting workflow. See logs for details.") from e
