#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import os
import sys
from pathlib import Path
from urllib.parse import quote

import pkginfo
import requests

NAT_ARCH = "any"
NAT_OS = "any"
NAT_COMPONENT = "nvidia-nat"
ARTIFACTORY_COMPONENT_FIXED_NAME = "aiqtoolkit"
EXCLUDE_SUBDIRS = {"examples"}


def upload_wheel(
    wheel_file: Path,
    wheels_base_dir: Path,
    artifactory_url: str,
    artifactory_name: str,
    username: str,
    api_key: str,
    git_tag: str,
    release_approver: str,
) -> str:
    relative_path = wheel_file.relative_to(wheels_base_dir)
    relative_path = Path(ARTIFACTORY_COMPONENT_FIXED_NAME, *relative_path.parts[1:])
    artifact_path = f"{artifactory_name}/{relative_path.as_posix()}"

    properties = {
        "arch": NAT_ARCH,
        "os": NAT_OS,
        "branch": git_tag,
        "component_name": ARTIFACTORY_COMPONENT_FIXED_NAME,
        "version": git_tag,
        "release_approver": release_approver,
    }
    matrix_parameters = "".join(f";{key}={quote(value, safe='')}" for key, value in properties.items())
    wheel_url = f"{artifactory_url.rstrip('/')}/{quote(artifact_path, safe='/')}"
    upload_url = f"{wheel_url}{matrix_parameters}"

    print(f"Uploading {wheel_file} to {artifact_path}...", flush=True)
    with wheel_file.open("rb") as wheel_data:
        response = requests.put(
            upload_url,
            data=wheel_data,
            auth=(username, api_key),
            timeout=(30, 600),
        )
    response.raise_for_status()
    return wheel_url


def perform_release(published_wheels: list[tuple[Path, str]]) -> None:
    kitmaker_url = os.environ["KITMAKER_URL"]
    kitmaker_api_token = os.environ["KITMAKER_API_TOKEN"]
    kitmaker_owner = os.environ["KITMAKER_OWNER"]
    headers = {"Authorization": f"Bearer {kitmaker_api_token}"}

    response = requests.get(
        f"{kitmaker_url}/api/v0/projects",
        headers=headers,
        timeout=(30, 600),
    )
    response.raise_for_status()
    projects = response.json()

    if len(projects) < len(published_wheels):
        print(
            f"Warning: KitMaker returned {len(projects)} projects for {len(published_wheels)} published wheels.",
            flush=True,
        )

    project_ids = {project["name"]: project["id"] for project in projects}
    for wheel_file, wheel_url in published_wheels:
        package_name = pkginfo.Wheel(str(wheel_file)).name
        package_id = project_ids[package_name]
        payload = {
            "project_name": package_name,
            "payload": [{
                "pic": kitmaker_owner,
                "job_type": "wheel-release-job",
                "url": wheel_url,
                "upload": True,
            }],
        }

        response = requests.post(
            f"{kitmaker_url}/api/v0/projects/{package_id}/releases",
            headers=headers,
            json=payload,
            timeout=(30, 600),
        )
        response.raise_for_status()


def main() -> int:
    project_dir = Path(os.environ["CI_PROJECT_DIR"])
    wheels_base_dir = project_dir / ".tmp" / "wheels"
    artifactory_url = os.environ["NAT_ARTIFACTORY_URL"]
    artifactory_name = os.environ["NAT_ARTIFACTORY_NAME"]
    username = os.environ["URM_USER"]
    api_key = os.environ["URM_API_KEY"]
    release_approver = os.environ["RELEASE_APPROVER"]
    git_tag = os.environ["GIT_TAG"]

    wheels = []
    published_wheels: list[tuple[Path, str]] = []

    wheels_dir = wheels_base_dir / NAT_COMPONENT
    print(f"Dir : {wheels_dir}", flush=True)

    for subdir in (path for path in wheels_dir.iterdir() if path.is_dir()):
        if subdir.name in EXCLUDE_SUBDIRS:
            print(f"Skipping excluded directory: {subdir.name}", flush=True)
            continue

        print(f"Uploading wheels from {subdir.relative_to(wheels_base_dir)} to Artifactory...", flush=True)
        for wheel_file in subdir.rglob("*.whl"):
            wheels.append(wheel_file)
            try:
                wheel_url = upload_wheel(
                    wheel_file,
                    wheels_base_dir,
                    artifactory_url,
                    artifactory_name,
                    username,
                    api_key,
                    git_tag,
                    release_approver,
                )
                published_wheels.append((wheel_file, wheel_url))
            except Exception as e:
                print(f"Failed to upload {wheel_file}: {e}", flush=True)

    num_unpublished = len(wheels) - len(published_wheels)
    if num_unpublished > 0:
        print(f"Warning: Only {len(published_wheels)} out of {len(wheels)} wheels were uploaded successfully.",
              flush=True)
    else:
        print("All wheels uploaded to Artifactory.")

    if (os.environ.get("CI_CRON_NIGHTLY") == "1" or os.environ.get("IS_TAGGED") == "1"
            or os.environ.get("CI_COMMIT_BRANCH") == "main"):
        print("Performing release of published wheels to KitMaker...", flush=True)
        perform_release(published_wheels)
    else:
        print("Skipping release to KitMaker. This is not a nightly, tagged, or main branch build.", flush=True)

    return num_unpublished


if __name__ == "__main__":
    sys.exit(main())
