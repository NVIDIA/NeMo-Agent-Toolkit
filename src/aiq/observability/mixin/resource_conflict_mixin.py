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
import weakref
from abc import ABC
from abc import abstractmethod
from typing import Any

logger = logging.getLogger(__name__)


class ResourceConflictError(ValueError):
    """Raised when multiple exporter instances would conflict over the same resource."""
    pass


class ResourceConflictMixin(ABC):
    """Abstract mixin for detecting resource conflicts between exporter instances.

    This mixin provides a framework for exporters to detect when multiple instances
    would conflict over the same resources (files, database tables, API endpoints, etc.).
    Each concrete implementation defines what constitutes a resource conflict for that
    exporter type.

    The mixin maintains class-level registries using weakrefs for automatic cleanup
    when instances are garbage collected.
    """

    # Each subclass gets its own registry - prevents cross-contamination
    _registries: dict[type, dict[str, weakref.ref]] = {}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Register this instance's resources and check for conflicts
        self._register_resources()

    @abstractmethod
    def _get_resource_identifiers(self) -> dict[str, Any]:
        """Return dict of resource_type -> identifier that this instance will use.

        Examples:
            Files: {"file_path": "/logs/app.log", "cleanup_pattern": "app_*.log"}
            Phoenix: {"project_endpoint": "my_project@http://localhost:6006"}
            Database: {"table_name": "events", "connection": "postgresql://..."}

        Returns:
            dict[str, Any]: Dict mapping resource type names to unique identifiers for those resources.
        """
        pass

    @abstractmethod
    def _format_conflict_error(self, resource_type: str, identifier: Any, existing_instance: Any) -> str:
        """Format a user-friendly error message for a resource conflict.

        Args:
            resource_type (str): The type of resource that conflicts (e.g., "file_path", "project_endpoint")
            identifier (Any): The identifier for this resource
            existing_instance (Any): The existing instance that conflicts with this one

        Returns:
            A clear error message explaining the conflict and how to resolve it.
        """
        pass

    def _register_resources(self):
        """Register this instance's resources and check for conflicts.

        Raises:
            ResourceConflictError: If any resource conflicts with an existing instance.
        """
        # Get our class-specific registry
        cls = type(self)
        if cls not in self._registries:
            self._registries[cls] = {}
        registry = self._registries[cls]

        # Clean up dead references first
        self._cleanup_dead_references(registry)

        # Check each resource for conflicts
        resources = self._get_resource_identifiers()
        for resource_type, identifier in resources.items():
            resource_key = f"{resource_type}:{identifier}"

            # Check for existing instance using this resource
            if resource_key in registry:
                existing_ref = registry[resource_key]
                existing_instance = existing_ref()
                if existing_instance is not None:
                    error_msg = self._format_conflict_error(resource_type, identifier, existing_instance)
                    raise ResourceConflictError(error_msg)

            # Register this instance for this resource
            registry[resource_key] = weakref.ref(self, lambda ref, key=resource_key: registry.pop(key, None))

        logger.debug("Registered %d resources for %s", len(resources), self.__class__.__name__)

    def _cleanup_dead_references(self, registry: dict[str, weakref.ref]):
        """Remove dead weakref entries from the registry.

        Args:
            registry (dict[str, weakref.ref]): The registry to clean up.
        """
        dead_keys = [key for key, ref in registry.items() if ref() is None]
        for key in dead_keys:
            registry.pop(key, None)

    @classmethod
    def get_active_resource_count(cls) -> int:
        """Get the number of active resources registered for this class.

        Returns:
            int: Number of active resource registrations.
        """
        if cls not in cls._registries:
            return 0

        registry = cls._registries[cls]
        # Clean up and count live references
        live_count = sum(1 for ref in registry.values() if ref() is not None)
        return live_count
