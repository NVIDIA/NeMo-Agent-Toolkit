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

import importlib.metadata
import inspect
import json
import logging
import typing
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import BaseModel
from pydantic import field_validator

from aiq.builder.framework_enum import LLMFrameworkEnum
from aiq.data_models.component import AIQComponentEnum
from aiq.utils.metadata_utils import generate_config_type_docs

if TYPE_CHECKING:
    from aiq.cli.type_registry import ToolWrapperBuildCallableT
    from aiq.data_models.common import TypedBaseModelT

logger = logging.getLogger(__name__)


class DiscoveryStatusEnum(str, Enum):
    SUCCESS = "success"
    FAILURE = "failure"


class DiscoveryContractFieldsEnum(str, Enum):
    PACKAGE = "package"
    VERSION = "version"
    COMPONENT_TYPE = "component_type"
    COMPONENT_NAME = "component_name"
    DESCRIPTION = "description"
    DEVELOPER_NOTES = "developer_notes"


class DiscoveryMetadata(BaseModel):
    """A data model representing metadata about each registered component to faciliate its discovery.

    Args:
        package (str): The name of the package containing the AgentIQ component.
        version (str): The version number of the package containing the AgentIQ component.
        component_type (AIQComponentEnum): The type of AgentIQ component this metadata represents.
        component_name (str): The registered name of the AgentIQ component.
        description (str): Description of the AgentIQ component pulled from its config objects docstrings.
        developer_notes (str): Other notes to a developers to aid in the use of the component.
        status (DiscoveryStatusEnum): Provides the status of the metadata discovery process.
    """

    package: str = ""
    version: str = ""
    component_type: AIQComponentEnum = AIQComponentEnum.UNDEFINED
    component_name: str = ""
    description: str = ""
    developer_notes: str = ""
    status: DiscoveryStatusEnum = DiscoveryStatusEnum.SUCCESS

    @field_validator("description", mode="before")
    @classmethod
    def ensure_description_string(cls, v: typing.Any):
        if not isinstance(v, str):
            return ""
        return v

    @staticmethod
    def get_preferred_item(items: list, preferred: str) -> str:
        return preferred if preferred in items else items[0]

    @staticmethod
    @lru_cache
    def get_distribution_name_from_metadata(root_package_name: str) -> str | None:
        """
        This is not performant and is only present to be used (not used
        currently) as a fallback when the distro name doesn't match the
        module name and private_data is not available to map it.
        """
        mapping = importlib.metadata.packages_distributions()
        try:
            distro_names = mapping.get(root_package_name, [None])
            distro_name = DiscoveryMetadata.get_preferred_item(distro_names, "agentiq")
        except KeyError:
            return root_package_name

        return distro_name if distro_name else root_package_name

    @staticmethod
    @lru_cache
    def get_distribution_name_from_private_data(root_package: str) -> str | None:
        # Locate distibution mapping stored in the packages private data
        module = __import__(root_package)
        for path in module.__path__:
            package_dir = Path(path).resolve()
            distinfo_path = package_dir / "meta" / "module_to_distro.json"

            if distinfo_path.exists():
                with distinfo_path.open("r") as f:
                    data = json.load(f)
                    return data.get(root_package, None)
        return None

    @staticmethod
    @lru_cache
    def get_distribution_name(root_package: str) -> str:
        """
        The aiq library packages use a distro name 'agentiq[]' and
        root package name 'aiq'. They provide mapping in a metadata file
        for optimized installation.
        """
        distro_name = DiscoveryMetadata.get_distribution_name_from_private_data(root_package)
        return distro_name if distro_name else root_package

    @staticmethod
    def from_config_type(config_type: type["TypedBaseModelT"],
                         component_type: AIQComponentEnum = AIQComponentEnum.UNDEFINED) -> "DiscoveryMetadata":
        """Generates discovery metadata from an AgentIQ config object.

        Args:
            config_type (type[TypedBaseModelT]): A registered component's configuration object.
            component_type (AIQComponentEnum, optional): The type of the registered component. Defaults to
            AIQComponentEnum.UNDEFINED.

        Returns:
            DiscoveryMetadata: A an object containing component metadata to facilitate discovery and reuse.
        """

        try:
            module = inspect.getmodule(config_type)
            root_package: str = module.__package__.split(".")[0]
            distro_name = DiscoveryMetadata.get_distribution_name(root_package)

            if not distro_name:
                # raise an exception
                logger.error("Encountered issue getting distro_name for module %s", module.__name__)
                return DiscoveryMetadata(status=DiscoveryStatusEnum.FAILURE)

            version = importlib.metadata.version(distro_name) if distro_name != "" else ""
        except Exception as e:
            logger.exception("Encountered issue extracting module metadata for %s: %s", config_type, e, exc_info=True)
            return DiscoveryMetadata(status=DiscoveryStatusEnum.FAILURE)

        description = generate_config_type_docs(config_type=config_type)

        return DiscoveryMetadata(package=distro_name,
                                 version=version,
                                 component_type=component_type,
                                 component_name=config_type.static_type(),
                                 description=description)

    @staticmethod
    def from_fn_wrapper(fn: "ToolWrapperBuildCallableT",
                        wrapper_type: LLMFrameworkEnum | str,
                        component_type: AIQComponentEnum = AIQComponentEnum.TOOL_WRAPPER) -> "DiscoveryMetadata":
        """Generates discovery metadata from function with specified wrapper type.

        Args:
            fn (ToolWrapperBuildCallableT): A tool wrapper callable to source component metadata.
            wrapper_type (LLMFrameworkEnum): The wrapper to apply to the callable to faciliate inter-framwork
            interoperability.

            component_type (AIQComponentEnum, optional): The type of the registered component. Defaults to
            AIQComponentEnum.TOOL_WRAPPER.

        Returns:
            DiscoveryMetadata: A an object containing component metadata to facilitate discovery and reuse.
        """

        try:
            module = inspect.getmodule(fn)
            root_package: str = module.__package__.split(".")[0]
            root_package = DiscoveryMetadata.get_distribution_name(root_package)
            version = importlib.metadata.version(root_package) if root_package != "" else ""
        except Exception as e:
            logger.exception("Encountered issue extracting module metadata for %s: %s", fn, e, exc_info=True)
            return DiscoveryMetadata(status=DiscoveryStatusEnum.FAILURE)

        if isinstance(wrapper_type, LLMFrameworkEnum):
            wrapper_type = wrapper_type.value

        return DiscoveryMetadata(package=root_package,
                                 version=version,
                                 component_type=component_type,
                                 component_name=wrapper_type,
                                 description=fn.__doc__ or "")

    @staticmethod
    def from_package_name(package_name: str, package_version: str | None) -> "DiscoveryMetadata":
        """Generates discovery metadata from an installed package name.

        Args:
            package_name (str): The name of the AgentIQ plugin package containing registered components.
            package_version (str, optional): The version of the package, Defaults to None.

        Returns:
            DiscoveryMetadata: A an object containing component metadata to facilitate discovery and reuse.
        """

        try:
            package_name = DiscoveryMetadata.get_distribution_name(package_name)
            metadata = importlib.metadata.metadata(package_name)
            description = metadata.get("Summary", "")
            if (package_version is None):
                package_version = importlib.metadata.version(package_name)
        except Exception as e:
            logger.exception("Encountered issue extracting module metadata for %s: %s", package_name, e, exc_info=True)
            return DiscoveryMetadata(status=DiscoveryStatusEnum.FAILURE)

        return DiscoveryMetadata(package=package_name,
                                 version=package_version,
                                 component_type=AIQComponentEnum.PACKAGE,
                                 component_name=package_name,
                                 description=description)

    @staticmethod
    def from_provider_framework_map(
            config_type: type["TypedBaseModelT"],
            wrapper_type: LLMFrameworkEnum | str | None,
            provider_type: AIQComponentEnum,
            component_type: AIQComponentEnum = AIQComponentEnum.UNDEFINED) -> "DiscoveryMetadata":
        """Generates discovery metadata from provider and framework mapping information.

        Args:
            config_type (type[TypedBaseModelT]): A registered component's configuration object.
            wrapper_type (LLMFrameworkEnum | str): The wrapper to apply to the callable to faciliate inter-framwork
            interoperability.

            provider_type (AIQComponentEnum): The type of provider the registered component supports.
            component_type (AIQComponentEnum, optional): The type of the registered component. Defaults to
            AIQComponentEnum.UNDEFINED.

        Returns:
            DiscoveryMetadata: A an object containing component metadata to facilitate discovery and reuse.
        """

        try:
            module = inspect.getmodule(config_type)
            root_package: str = module.__package__.split(".")[0]
            root_package = DiscoveryMetadata.get_distribution_name(root_package)
            version = importlib.metadata.version(root_package) if root_package != "" else ""
        except Exception as e:
            logger.exception("Encountered issue extracting module metadata for %s: %s", config_type, e, exc_info=True)
            return DiscoveryMetadata(status=DiscoveryStatusEnum.FAILURE)

        wrapper_type = wrapper_type.value if isinstance(wrapper_type, LLMFrameworkEnum) else wrapper_type
        component_name = f"{config_type.static_type()} ({provider_type.value}) - {wrapper_type}"

        description = generate_config_type_docs(config_type=config_type)

        return DiscoveryMetadata(package=root_package,
                                 version=version,
                                 component_type=component_type,
                                 component_name=component_name,
                                 description=description)
