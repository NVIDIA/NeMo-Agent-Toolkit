#!/usr/bin/env python
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

import click
import tomlkit
import tomlkit.items
from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet


@click.command()
@click.option("--toml-file-path", required=True, type=click.Path(exists=True), help="Path to the TOML file.")
@click.option("--new-version", required=True, help="New version to set for the package.")
@click.option("--package-name", default="nvidia-nat", help="Name of the package to update.")
@click.option("--version-match", default="~=", help="Version match specifier to use for the dependency.")
def main(toml_file_path: str, new_version: str, package_name: str, version_match: str):
    """
    Update the dependency version of nvidia-nat that a plugin depends on in the pyproject.toml file.

    Parameters
    ----------
    toml_file_path : str
        Path to the TOML file.
    new_version : str
        New version to set for the package.
    package_name : str
        Name of the package to update.
    version_match : str
        Version match specifier to use for the dependency.
    """
    with open(toml_file_path, "r", encoding="utf-8") as fh:
        toml_data = tomlkit.load(fh)

    toml_project: tomlkit.items.Table = toml_data['project']
    depdendencies: tomlkit.items.Array = toml_project['dependencies']
    for (i, dep) in enumerate(depdendencies):
        req = Requirement(dep)
        if req.name == package_name:  # will also match nvidia-nat[<plugin>]
            # Update the version specifier
            specifier = SpecifierSet(f"{version_match}{new_version}")
            req.specifier = specifier
            depdendencies[i] = str(req)

    # Write the updated TOML file
    with open(toml_file_path, "w", encoding="utf-8") as fh:
        tomlkit.dump(toml_data, fh)


if __name__ == "__main__":
    main()
