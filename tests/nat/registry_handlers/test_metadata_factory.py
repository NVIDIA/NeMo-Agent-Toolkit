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

from nat.cli.type_registry import TypeRegistry
from nat.data_models.component import ComponentEnum
from nat.registry_handlers.metadata_factory import ComponentDiscoveryMetadata
from nat.registry_handlers.package_utils import build_wheel
from nat.registry_handlers.schemas.package import WheelData
from nat.runtime.loader import PluginTypes
from nat.runtime.loader import discover_and_register_plugins


@pytest.mark.parametrize("use_wheel_data", [
    (True),
    (False),
])
def test_metadata_factory(registry: TypeRegistry, use_wheel_data: bool):

    discover_and_register_plugins(PluginTypes.CONFIG_OBJECT)

    package_root = "."

    wheel_data: WheelData | None = None
    if (use_wheel_data):
        wheel_data = build_wheel(package_root=package_root)
        registry.register_package(package_name=wheel_data.package_name, package_version=wheel_data.whl_version)

    for component_type in [ComponentEnum.PACKAGE]:
        if component_type == ComponentEnum.UNDEFINED:
            continue
        component_discovery_metadata = ComponentDiscoveryMetadata.from_package_component_type(
            component_type=component_type, wheel_data=wheel_data)

        component_discovery_metadata.load_metadata()
        component_metadata_items = component_discovery_metadata.get_metadata_items()

        if (wheel_data is not None):
            assert len(component_metadata_items) > 0
        else:
            if (component_type == ComponentEnum.PACKAGE):
                assert len(component_metadata_items) == 0
            else:
                assert len(component_metadata_items) > 0

        for metadata_item in component_metadata_items:
            assert metadata_item["status"] == "success"
            assert metadata_item["component_type"] == component_type
