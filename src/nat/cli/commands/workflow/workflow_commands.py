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

import json
import logging
import os
import shutil
import subprocess
from importlib.metadata import Distribution, PackageNotFoundError
from pathlib import Path
from urllib.parse import urlparse

import click
from jinja2 import Environment, FileSystemLoader

logger = logging.getLogger(__name__)


def _get_nat_dependency(versioned: bool = True) -> str:
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
        "and describe the component when inspecting with 'nat info workflow'."
    ),
)
def create_command(workflow_name: str, install: bool, workflow_dir: str, description: str) -> None:
    if not workflow_name or not workflow_name.strip():
        raise click.BadParameter("Workflow name cannot be empty.")

    # Block absolute paths, traversal, and separator characters
    if Path(workflow_name).is_absolute() or workflow_name in {".", ".."}:
        raise click.BadParameter("Workflow name cannot be a path or special name.")
    seps = tuple(s for s in (os.sep, os.altsep) if s)
    if any(sep in workflow_name for sep in seps):
        raise click.BadParameter("Workflow name cannot contain path separators.")

    try:
        try:
            repo_root = get_repo_root()
        except PackageError:
            repo_root = None

        workflow_dir = Path(workflow_dir).resolve()
        if not workflow_dir.exists():
            raise click.BadParameter(f"Invalid workflow directory: {workflow_dir}")

        template_dir = Path(__file__).parent / "templates"
        new_workflow_dir = workflow_dir / workflow_name
        if not new_workflow_dir.resolve().is_relative_to(workflow_dir):
            raise click.BadParameter("Workflow name resolves outside the output directory.")

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

        # Jinja2 environment for non-HTML templates
        env = Environment(loader=FileSystemLoader(str(template_dir)), autoescape=False)
        editable = repo_root is not None
        uv_available = shutil.which("uv") is not None
        use_uv = editable and uv_available

        install_cmd = (
            ["uv", "pip", "install", "-e", str(new_workflow_dir)]
            if use_uv
            else ["pip", "install", "-e", str(new_workflow_dir)]
        )

        files_to_render = {
            "pyproject.toml.j2": new_workflow_dir / "pyproject.toml",
            "register.py.j2": base_dir / "register.py",
            "workflow.py.j2": base_dir / f"{package_name}_function.py",
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

        # Symlink fallback for configs/data
        for src, name in ((configs_dir, "configs"), (data_dir, "data")):
            dest = new_workflow_dir / name
            try:
                os.symlink(src, dest)
            except OSError:
                if dest.exists():
                    if dest.is_symlink():
                        dest.unlink()
                    else:
                        shutil.rmtree(dest)
                shutil.copytree(src, dest)

        if install:
            click.echo(f"Installing workflow '{workflow_name}'...")
            result = subprocess.run(install_cmd, capture_output=True, text=True, check=False)
            if result.returncode != 0:
                click.echo(f"An error occurred during installation:\n{result.stderr}")
                return
            click.echo(f"Workflow '{workflow_name}' installed successfully.")

        click.echo(f"Workflow '{workflow_name}' created successfully in '{new_workflow_dir}'.")

    except Exception:
        logger.exception("Error while creating workflow")
        click.echo(f"An error occurred while creating the workflow: {workflow_name}")


@click.command()
@click.argument("workflow_name")
def reinstall_command(workflow_name: str) -> None:
    try:
        editable = get_repo_root() is not None
        workflow_dir = get_workflow_path_from_name(workflow_name)
        if not workflow_dir or not workflow_dir.exists():
            click.echo(f"Workflow '{workflow_name}' does not exist.")
            return

        uv_available = shutil.which("uv") is not None
        use_uv = editable and uv_available
        reinstall_cmd = (
            ["uv", "pip", "install", "-e", str(workflow_dir)]
            if use_uv
            else ["pip", "install", "-e", str(workflow_dir)]
        )

        click.echo(f"Reinstalling workflow '{workflow_name}'...")
        result = subprocess.run(reinstall_cmd, capture_output=True, text=True, check=False)
        if result.returncode != 0:
            click.echo(f"An error occurred during installation:\n{result.stderr}")
            return

        click.echo(f"Workflow '{workflow_name}' reinstalled successfully.")
    except Exception:
        logger.exception("Error while reinstalling workflow")
        click.echo(f"An error occurred while reinstalling the workflow: {workflow_name}")


@click.command()
@click.argument("workflow_name")
def delete_command(workflow_name: str) -> None:
    try:
        if not click.confirm(f"Are you sure you want to delete the workflow '{workflow_name}'?"):
            click.echo("Workflow deletion cancelled.")
            return

        editable = get_repo_root() is not None
        workflow_dir = get_workflow_path_from_name(workflow_name)
        package_name = _get_module_name(workflow_name)

        uv_available = shutil.which("uv") is not None
        use_uv = editable and uv_available
        uninstall_cmd = (
            ["uv", "pip", "uninstall", "-y", package_name]
            if use_uv
            else ["pip", "uninstall", "-y", package_name]
        )

        click.echo(f"Uninstalling workflow '{workflow_name}' package...")
        result = subprocess.run(uninstall_cmd, capture_output=True, text=True, check=False)
        if result.returncode != 0:
            click.echo(f"An error occurred during uninstallation:\n{result.stderr}")
            return

        click.echo(f"Workflow '{workflow_name}' (package '{package_name}') successfully uninstalled")

        if not workflow_dir or not workflow_dir.exists():
            click.echo(f"Unable to locate local files for {workflow_name}. Nothing will be deleted.")
            return

        click.echo(f"Deleting workflow directory '{workflow_dir}'...")
        shutil.rmtree(workflow_dir)
        click.echo(f"Workflow '{workflow_name}' deleted successfully.")

    except Exception:
        logger.exception("Error while deleting workflow")
        click.echo(f"An error occurred while deleting the workflow: {workflow_name}")


# Compatibility alias
AIQPackageError = PackageError
