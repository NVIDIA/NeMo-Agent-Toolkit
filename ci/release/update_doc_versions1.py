#!/usr/bin/env python
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

import json

import click


@click.command()
@click.option("--versions-file", required=True, type=click.Path(exists=True), help="Path to the versions file.")
@click.option("--new-version", required=True, help="New version to set for the package.")
def main(versions_file: str, new_version: str):
    if new_version.count('.') != 1:
        raise ValueError("Version string must only include <major>.<minor>")

    with open(versions_file, "r", encoding="utf-8") as fh:
        version_list = json.load(fh)

    for version_data in version_list:
        version_data.pop('preferred', None)
        if version_data['version'] == new_version:
            raise ValueError(f"Version {new_version} already exists in the versions file.")

    version_list.insert(0, {"version": new_version, "preferred": True, "url": f"../{new_version}/"})

    with open(versions_file, "w", encoding="utf-8") as fh:
        json.dump(version_list, fh, indent=4)
        fh.write("\n")  # Add a trailing newline


if __name__ == "__main__":
    main()
