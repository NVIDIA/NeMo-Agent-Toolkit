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

import logging

from aiq.data_models.component import AIQComponentEnum
from aiq.data_models.discovery_metadata import DiscoveryMetadata
from aiq.data_models.discovery_metadata import DiscoveryStatusEnum
from aiq.registry_handlers.schemas.package import WheelData

logger = logging.getLogger(__name__)


class ComponentDiscoveryMetadata:

    def __init__(self, component_type: AIQComponentEnum, wheel_data: WheelData | None = None):
        self._component_type = component_type
        self._metadata_items: list[dict | DiscoveryMetadata] = []
        self._wheel_data: WheelData = wheel_data

    def load_metadata(self):

        from aiq.cli.type_registry import GlobalTypeRegistry

        registry = GlobalTypeRegistry.get()

        for _, registered_component_info in registry.get_infos_by_type(component_type=self._component_type).items():

            if ((registered_component_info.discovery_metadata.status == DiscoveryStatusEnum.SUCCESS) and
                ((self._wheel_data is None) or
                 (registered_component_info.discovery_metadata.package in self._wheel_data.union_dependencies))):

                if ((self._wheel_data is not None)
                        and (registered_component_info.discovery_metadata.package == self._wheel_data.package_name)):
                    discovery_metadata_copy = registered_component_info.discovery_metadata.model_copy(deep=True)
                    discovery_metadata_copy.version = self._wheel_data.whl_version
                    self._metadata_items.append(discovery_metadata_copy.model_dump())
                    continue

                self._metadata_items.append(registered_component_info.discovery_metadata.model_dump())

    def get_metadata_items(self) -> list[dict | DiscoveryMetadata]:
        return self._metadata_items

    @staticmethod
    def from_package_component_type(component_type: AIQComponentEnum,
                                    wheel_data: WheelData | None = None) -> "ComponentDiscoveryMetadata":
        return ComponentDiscoveryMetadata(component_type=component_type, wheel_data=wheel_data)
