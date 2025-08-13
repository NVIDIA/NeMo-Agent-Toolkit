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

import logging
import os.path
import shutil
import subprocess
from pathlib import Path
from urllib.parse import urlparse

import click
from jinja2 import Environment
from jinja2 import FileSystemLoader

logger = logging.getLogger(__name__)


class PackageError(Exception):
    pass


def get_repo_root():
    return find_package_root("nvidia-nat")


def _get_module_name(workflow_name: str):
    return workflow_name.replace("-", "_")


def _generate_valid_classname(class_name: str):
    return class_name.replace('_', ' ').replace('-', ' ').title().replace(' ', '')


def find_package_root(package_name: str) -> Path | None:
    """
    Find the root directory for a python package installed with the "editable" option.

    Args:
        package_name: The python package name as it appears when importing it into a python script

    Returns:
        Posix path pointing to the package root
    """
    import json
    from importlib.metadata import Distribution
    from importlib.metadata import PackageNotFoundError

    try:
        dist_info = Distribution.from_name(package_name)
        direct_url = dist_info.read_text("direct_url.json")
        if not direct_url:
            return None

        try:
            info = json.loads(direct_url)
        except json.JSONDecodeError:
            logger.error("Malformed direct_url.json for package: %s", package_name)
            return None

        if not info.get("dir_info", {}).get("editable"):
            return None

        # Parse URL
        url = info.get("url", "")
        parsed_url = urlparse(url)

        if parsed_url.scheme != "file":
            logger.error("Invalid URL scheme in direct_url.json: %s", url)
            return None

        package_root = Path(parsed_url.path).resolve()

        # Ensure the path exists and is within an allowed base directory
        if not package_root.exists() or not package_root.is_dir():
            logger.error("Package root does not exist: %s", package_root)
            return None

        return package_root

    except TypeError:
        return None

    except PackageNotFoundError as e:
        raise PackageError(f"Package {package_name} is not installed") from e


def get_workflow_path_from_name(workflow_name: str):
    """
    Look up the location of an installed AIQ Toolkit workflow and retrieve the root directory of the installed workflow.

    Args:
        workflow_name: The name of the workflow.

    Returns:
        Path object for the workflow's root directory.
    """
    # Get the module name as a valid package name.
    try:
        module_name = _get_module_name(workflow_name)
        package_root = find_package_root(module_name)
        return package_root

    except PackageError as e:
        logger.info("Unable to get the directory path for %s: %s", workflow_name, e)
        return None


@click.command()
@click.argument('workflow_name')
@click.option('--install/--no-install', default=True, help="Whether to install the workflow package immediately.")
@click.option(
    "--workflow-dir",
    default=".",
    help="Output directory for saving the created workflow. A new folder with the workflow name will be created "
    "within. Defaults to the present working directory.")
@click.option(
    "--description",
    default="AIQ Toolkit function template. Please update the description.",
    help="""A description of the component being created. Will be used to populate the docstring and will describe the
         component when inspecting installed components using 'aiq info component'""")
# pylint: disable=missing-param-doc
def create_command(workflow_name: str, install: bool, workflow_dir: str, description: str):
    """
    Create a new AIQ Toolkit workflow using templates.

    Args:
        workflow_name (str): The name of the new workflow.
        install (bool): Whether to install the workflow package immediately.
        workflow_dir (str): The directory to create the workflow package.
        description (str): Description to pre-popluate the workflow docstring.
    """
    try:
        # Get the repository root
        try:
            repo_root = get_repo_root()
        except PackageError:
            repo_root = None

        # Get the absolute path for the output directory
        if not os.path.isabs(workflow_dir):
            workflow_dir = os.path.abspath(workflow_dir)

        if not os.path.exists(workflow_dir):
            raise ValueError(f"Invalid workflow directory specified. {workflow_dir} does not exist.")

        # Define paths
        template_dir = Path(__file__).parent / 'templates'
        new_workflow_dir = Path(workflow_dir) / workflow_name
        package_name = _get_module_name(workflow_name)
        rel_path_to_repo_root = "" if not repo_root else os.path.relpath(repo_root, new_workflow_dir)

        # Check if the workflow already exists
        if new_workflow_dir.exists():
            click.echo(f"Workflow '{workflow_name}' already exists.")
            return

        # Create directory structure
        (new_workflow_dir / 'src' / package_name).mkdir(parents=True)
        # Create config directory
        (new_workflow_dir / 'src' / package_name / 'configs').mkdir(parents=True)
        # Create package level configs directory
        (new_workflow_dir / 'configs').mkdir(parents=True)

        # Initialize Jinja2 environment
        env = Environment(loader=FileSystemLoader(str(template_dir)))
        editable = get_repo_root() is not None

        if editable:
            install_cmd = ['uv', 'pip', 'install', '-e', str(new_workflow_dir)]
        else:
            install_cmd = ['pip', 'install', '-e', str(new_workflow_dir)]

        # List of templates and their destinations
        files_to_render = {
            'pyproject.toml.j2': new_workflow_dir / 'pyproject.toml',
            'register.py.j2': new_workflow_dir / 'src' / package_name / 'register.py',
            'workflow.py.j2': new_workflow_dir / 'src' / package_name / f'{workflow_name}_function.py',
            '__init__.py.j2': new_workflow_dir / 'src' / package_name / '__init__.py',
            'config.yml.j2': new_workflow_dir / 'src' / package_name / 'configs' / 'config.yml',
        }

        # Render templates
        context = {
            'editable': editable,
            'workflow_name': workflow_name,
            'python_safe_workflow_name': workflow_name.replace("-", "_"),
            'package_name': package_name,
            'rel_path_to_repo_root': rel_path_to_repo_root,
            'workflow_class_name': f"{_generate_valid_classname(workflow_name)}FunctionConfig",
            'workflow_description': description
        }

        for template_name, output_path in files_to_render.items():
            template = env.get_template(template_name)
            content = template.render(context)
            with open(output_path, 'w', encoding="utf-8") as f:
                f.write(content)

        # Create symlink for config.yml
        config_source = new_workflow_dir / 'src' / package_name / 'configs' / 'config.yml'
        config_link = new_workflow_dir / 'configs' / 'config.yml'
        os.symlink(config_source, config_link)

        if install:
            # Install the new package without changing directories
            click.echo(f"Installing workflow '{workflow_name}'...")
            result = subprocess.run(install_cmd, capture_output=True, text=True, check=True)

            if result.returncode != 0:
                click.echo(f"An error occurred during installation:\n{result.stderr}")
                return

            click.echo(f"Workflow '{workflow_name}' installed successfully.")

        click.echo(f"Workflow '{workflow_name}' created successfully in '{new_workflow_dir}'.")
    except Exception as e:
        logger.exception("An error occurred while creating the workflow: %s", e, exc_info=True)
        click.echo(f"An error occurred while creating the workflow: {e}")


@click.command()
@click.argument('workflow_name')
def reinstall_command(workflow_name):
    """
    Reinstall an AIQ Toolkit workflow to update dependencies and code changes.

    Args:
        workflow_name (str): The name of the workflow to reinstall.
    """
    try:
        editable = get_repo_root() is not None

        workflow_dir = get_workflow_path_from_name(workflow_name)
        if not workflow_dir or not workflow_dir.exists():
            click.echo(f"Workflow '{workflow_name}' does not exist.")
            return

        # Reinstall the package without changing directories
        click.echo(f"Reinstalling workflow '{workflow_name}'...")
        if editable:
            reinstall_cmd = ['uv', 'pip', 'install', '-e', str(workflow_dir)]
        else:
            reinstall_cmd = ['pip', 'install', '-e', str(workflow_dir)]

        result = subprocess.run(reinstall_cmd, capture_output=True, text=True, check=True)

        if result.returncode != 0:
            click.echo(f"An error occurred during installation:\n{result.stderr}")
            return

        click.echo(f"Workflow '{workflow_name}' reinstalled successfully.")
    except Exception as e:
        logger.exception("An error occurred while reinstalling the workflow: %s", e, exc_info=True)
        click.echo(f"An error occurred while reinstalling the workflow: {e}")


@click.command()
@click.argument('workflow_name')
def delete_command(workflow_name: str):
    """
    Delete an AIQ Toolkit workflow and uninstall its package.

    Args:
        workflow_name (str): The name of the workflow to delete.
    """
    try:
        if not click.confirm(f"Are you sure you want to delete the workflow '{workflow_name}'?"):
            click.echo("Workflow deletion cancelled.")
            return
        editable = get_repo_root() is not None

        workflow_dir = get_workflow_path_from_name(workflow_name)
        package_name = _get_module_name(workflow_name)

        if editable:
            uninstall_cmd = ['uv', 'pip', 'uninstall', package_name]
        else:
            uninstall_cmd = ['pip', 'uninstall', '-y', package_name]

        # Uninstall the package
        click.echo(f"Uninstalling workflow '{workflow_name}' package...")
        result = subprocess.run(uninstall_cmd, capture_output=True, text=True, check=True)

        if result.returncode != 0:
            click.echo(f"An error occurred during uninstallation:\n{result.stderr}")
            return
        click.echo(
            f"Workflow '{workflow_name}' (package '{package_name}') successfully uninstalled from python environment")

        if not workflow_dir or not workflow_dir.exists():
            click.echo(f"Unable to locate local files for {workflow_name}. Nothing will be deleted.")
            return

        # Remove the workflow directory
        click.echo(f"Deleting workflow directory '{workflow_dir}'...")
        shutil.rmtree(workflow_dir)

        click.echo(f"Workflow '{workflow_name}' deleted successfully.")
    except Exception as e:
        logger.exception("An error occurred while deleting the workflow: %s", e, exc_info=True)
        click.echo(f"An error occurred while deleting the workflow: {e}")


# Compatibility aliases with previous releases
AIQPackageError = PackageError
