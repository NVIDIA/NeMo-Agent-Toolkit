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

import pytest

from aiq.data_models.component import AIQComponentEnum
from aiq.data_models.discovery_metadata import DiscoveryMetadata
from aiq.registry_handlers.package_utils import build_aiq_artifact
from aiq.registry_handlers.package_utils import build_package_metadata
from aiq.registry_handlers.package_utils import build_wheel
from aiq.registry_handlers.schemas.package import WheelData
from aiq.registry_handlers.schemas.publish import AIQArtifact


def test_build_wheel():

    package_root = "."

    wheel_data = build_wheel(package_root=package_root)

    assert isinstance(wheel_data, WheelData)
    assert wheel_data.package_root == package_root


@pytest.mark.parametrize("use_wheel_data", [
    (True),
    (False),
])
def test_build_package_metadata(use_wheel_data):

    wheel_data: WheelData | None = None
    if (use_wheel_data):
        wheel_data = WheelData(package_root=".",
                               package_name="aiq",
                               toml_project={},
                               toml_dependencies=set(),
                               toml_aiq_packages=set(),
                               union_dependencies=set(),
                               whl_path="whl/path.whl",
                               whl_base64="",
                               whl_version="")

    discovery_metadata = build_package_metadata(wheel_data=wheel_data)

    assert isinstance(discovery_metadata, dict)

    for component_type, discovery_metadatas in discovery_metadata.items():
        assert isinstance(component_type, AIQComponentEnum)

        for discovery_metadata in discovery_metadatas:
            DiscoveryMetadata(**discovery_metadata)


def test_build_aiq_artifact():

    package_root = "."

    aiq_artifact = build_aiq_artifact(package_root=package_root)

    assert isinstance(aiq_artifact, AIQArtifact)
