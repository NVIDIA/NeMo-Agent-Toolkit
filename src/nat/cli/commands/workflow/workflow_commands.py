# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS"
# BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,
# either express or implied. See the License for the specific language
# governing permissions and limitations under the License.

import json
import logging
import os
import shutil
import subprocess
from importlib.metadata import Distribution
from importlib.metadata import PackageNotFoundError
from pathlib import Path
from urllib.parse import urlparse

import click
from jinja2 import Environment, FileSystemLoader

logger = logging.getLogger(__name__)


def _get_nat_dependency(versioned: bool = True) -> str:
    """
    Get the NAT dependency string with version.

    Args:
        versioned: Whether to include the version in the dependency string.

    Returns:
        str: The dependency string to use in pyproject.toml.
    """
    dependency = "nvidia-nat[langchain]"

    if not versioned:
        logger.debug("Using unversioned NAT dependency: %s", dependency)
        return dependency

    from nat.cli.entrypoint import get_version

    current_version = get_version()
    if current_version == "unknown":
        logger.warning("Could not detect NAT version, using unversioned dependency")
        return dependency

    major_minor = ".".join(current_version.split(".")[:2])
    dependency += f"~={major_minor}"
    logger.debug("Using NAT dependency: %s", dependency)
    return dependency


class PackageError(Exception):
    """Custom exception raised when package lookup fails."""


def get_repo_root():
    return find_package_root("nvidia-nat")


def _get_module_name(workflow_name: str) -> str:
    return workflow_name.replace("-", "_")


def _generate_valid_classname(class_name: str) -> str:
    return class_name.replace("_", " ").replace("-", " ").title().replace(" ", "")


def find_package_root(package_name: str) -> Path | None:
    """
    Find the root directory for a Python package installed with "editable" mode.

    Args:
        package_name: The package name used in imports.

    Returns:
        Path: Root directory of the package, or None if not found.
    """
    try:
        dist_info = Distribution.from_name(package_name)
        direct_url = dist_info.read_text("direct_url.json")
        if not direct_url:
            return None

        try:
            info = json.loads(direct_url)
        except json.JSONDecodeError:
            logger.exception("Malformed direct_url.json for package: %s", package_name)
            return None

        if not info.get("dir_info", {}).get("editable"):
            return None

        parsed_url = urlparse(info.get("url", ""))
        if parsed_url.scheme != "file":
            logger.error("Invalid URL scheme in direct_url.json: %s", info.get("url"))
            return None

        package_root = Path(parsed_url.path).resolve()
        if not package_root.exists() or not package_root.is_dir():
            logger.error("Package root does not exist: %s", package_root)
            return None

        return package_root

    except TypeError:
        return None
    except PackageNotFoundError as e:
        raise PackageError(f"Package {package_name} is not installed") from e


def get_workflow_path_from_name(workflow_name: str) -> Path | None:
    """
    Retrieve the root directory of an installed workflow by name.

    Args:
        workflow_name: The name of the workflow.

    Returns:
        Path: The root directory of the workflow, or None if unavailable.
    """
    try:
        module_name = _get_module_name(workflow_name)
        return find_package_root(module_name)
    except PackageError as e:
        logger.info("Unable to get path for workflow %s: %s", workflow_name, e)
        return None


@click.command()
@click.argument("workflow_name")
@click.option(
    "--install/--no-install",
    default=True,
    help="Whether to install the workflow package immediately.",
)
@click.option(
    "--workflow-dir",
    default=".",
    help=(
        "Output directory for saving the created workflow. "
        "A new folder with the workflow name will be created within. "
        "Defaults to the present working directory."
    ),
)
@click.option(
    "--description",
    default="NAT function template. Please update the description.",
    help=(
        "Description of the component being created. Used to populate the docstring "
        "and describe the component when inspecting with 'nat info component'."
    ),
)
def create_command(workflow_name: str, install: bool, workflow_dir: str, description: str):
    """
    Create a new NAT workflow using templates.
    """
    if not workflow_name or not workflow_name.strip():
        raise click.BadParameter("Workflow name cannot be empty.")

    try:
        try:
            repo_root = get_repo_root()
        except PackageError:
            repo_root = None

        workflow_dir = os.path.abspath(workflow_dir)
        if not os.path.exists(workflow_dir):
            raise click.BadParameter(f"Invalid workflow directory: {workflow_dir}")

        template_dir = Path(__file__).parent / "templates"
        new_workflow_dir = Path(workflow_dir) / workflow_name
        package_name = _get_module_name(workflow_name)
        rel_path_to_repo_root = "" if not repo_root else os.path.relpath(repo_root, new_workflow_dir)

        if new_workflow_dir.exists():
            click.echo(f"Workflow '{workflow_name}' already exists.")
            return

        base_dir = new_workflow_dir / "src" / package_name
        configs_dir = base_dir / "configs"
        data_dir = base_dir / "data"

        base_dir.mkdir(parents=True)
        configs_dir.mkdir(parents=True)
        data_dir.mkdir(parents=True)

        env = Environment(loader=FileSystemLoader(str(template_dir)))
        editable = repo_root is not None

        install_cmd = (
            ["uv", "pip", "install", "-e", str(new_workflow_dir)]
            if editable
            else ["pip", "install", "-e", str(new_workflow_dir)]
        )

        files_to_render = {
            "pyproject.toml.j2": new_workflow_dir / "pyproject.toml",
            "register.py.j2": base_dir / "register.py",
            "workflow.py.j2": base_dir / f"{workflow_name}_function.py",
            "__init__.py.j2": base_dir / "__init__.py",
            "config.yml.j2": configs_dir / "config.yml",
        }

        context = {
            "editable": editable,
            "workflow_name": workflow_name,
            "python_safe_workflow_name": workflow_name.replace("-", "_"),
            "package_name": package_name,
            "rel_path_to_repo_root": rel_path_to_repo_root,
            "workflow_class_name": f"{_generate_valid_classname(workflow_name)}FunctionConfig",
            "workflow_description": description,
            "nat_dependency": _get_nat_dependency(),
        }

        for template_name, output_path in files_to_render.items():
            template = env.get_template(template_name)
            content = template.render(context)
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(content)

        try:
            os.symlink(configs_dir, new_workflow_dir / "configs")
            os.symlink(data_dir, new_workflow_dir / "data")
        except OSError:
            shutil.copytree(configs_dir, new_workflow_dir / "configs")
            shutil.copytree(data_dir, new_workflow_dir / "data")

        if install:
            click.echo(f"Installing workflow '{workflow_name}'...")
            result = subprocess.run(install_cmd, capture_output=True, text=True, check=True)

            if result.returncode != 0:
                click.echo(f"An error occurred during installation:\n{result.stderr}")
                return

            click.echo(f"Workflow '{workflow_name}' installed successfully.")

        click.echo(f"Workflow '{workflow_name}' created successfully in '{new_workflow_dir}'.")
    except Exception as e:
        logger.exception("Error while creating workflow: %s", e)
        click.echo(f"An error occurred while creating the workflow: {e}")


@click.command()
@click.argument("workflow_name")
def reinstall_command(workflow_name: str):
    """
    Reinstall a NAT workflow to update dependencies and code changes.
    """
    try:
        editable = get_repo_root() is not None
        workflow_dir = get_workflow_path_from_name(workflow_name)

        if not workflow_dir or not workflow_dir.exists():
            click.echo(f"Workflow '{workflow_name}' does not exist.")
            return

        reinstall_cmd = (
            ["uv", "pip", "install", "-e", str(workflow_dir)]
            if editable
            else ["pip", "install", "-e", str(workflow_dir)]
        )

        click.echo(f"Reinstalling workflow '{workflow_name}'...")
        result = subprocess.run(reinstall_cmd, capture_output=True, text=True, check=True)

        if result.returncode != 0:
            click.echo(f"An error occurred during installation:\n{result.stderr}")
            return

        click.echo(f"Workflow '{workflow_name}' reinstalled successfully.")
    except Exception as e:
        logger.exception("Error while reinstalling workflow: %s", e)
        click.echo(f"An error occurred while reinstalling the workflow: {e}")


@click.command()
@click.argument("workflow_name")
def delete_command(workflow_name: str):
    """
    Delete a NAT workflow and uninstall its package.
    """
    try:
        if not click.confirm(f"Are you sure you want to delete the workflow '{workflow_name}'?"):
            click.echo("Workflow deletion cancelled.")
            return

        editable = get_repo_root() is not None
        workflow_dir = get_workflow_path_from_name(workflow_name)
        package_name = _get_module_name(workflow_name)

        uninstall_cmd = (
            ["uv", "pip", "uninstall", package_name]
            if editable
            else ["pip", "uninstall", "-y", package_name]
        )

        click.echo(f"Uninstalling workflow '{workflow_name}' package...")
        result = subprocess.run(uninstall_cmd, capture_output=True, text=True, check=True)

        if result.returncode != 0:
            click.echo(f"An error occurred during uninstallation:\n{result.stderr}")
            return

        click.echo(
            f"Workflow '{workflow_name}' (package '{package_name}') successfully "
            "uninstalled from Python environment"
        )

        if not workflow_dir or not workflow_dir.exists():
            click.echo(f"Unable to locate local files for {workflow_name}. Nothing will be deleted.")
            return

        click.echo(f"Deleting workflow directory '{workflow_dir}'...")
        shutil.rmtree(workflow_dir)
        click.echo(f"Workflow '{workflow_name}' deleted successfully.")
    except Exception as e:
        logger.exception("Error while deleting workflow: %s", e)
        click.echo(f"An error occurred while deleting the workflow: {e}")


# Compatibility alias
AIQPackageError = PackageError
