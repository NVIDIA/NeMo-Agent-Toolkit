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

import asyncio
import logging
import threading
from collections.abc import Awaitable
from collections.abc import Callable
from typing import TypeAlias

from aiq.observability.base_exporter import AbstractExporter
from aiq.observability.span.span_publisher import SpanPublisher

logger = logging.getLogger(__name__)

ExporterFactory: TypeAlias = Callable[[], Awaitable[AbstractExporter]]


class ExporterRegistry:
    """
    Singleton registry for managing exporter factories used in observability.
    Provides thread-safe instantiation and async-safe operations for adding,
    removing, and retrieving exporters.

    The registry stores exporter factories instead of instances, allowing for
    creation of new exporter instances for each request.
    """
    _instance = None
    _instance_lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        """
        Create or return the singleton instance of ExporterRegistry.
        Ensures thread-safe instantiation.
        """
        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    @classmethod
    async def _create_span_publisher(cls) -> AbstractExporter:
        """Create a new SpanPublisher instance for exporting spans.

        Returns:
            AbstractExporter: A new SpanPublisher instance that implements the span export functionality.
        """
        return SpanPublisher()

    def __init__(self):
        """
        Initialize the ExporterRegistry with a default exporter factory and an asyncio lock.
        This method is only called once for the singleton instance.
        """
        if getattr(self, '_initialized', False):
            return
        # The default exporter is the generic span publisher
        self._exporter_factories: dict[str, ExporterFactory] = {"span_publisher": self._create_span_publisher}
        self._lock = asyncio.Lock()
        self._initialized = True

    @classmethod
    def get_instance(cls) -> "ExporterRegistry":
        """
        Get the singleton instance of ExporterRegistry.

        Returns:
            ExporterRegistry: The singleton instance.
        """
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    async def add(self, name: str, exporter_factory: ExporterFactory) -> None:
        """
        Add a new exporter factory to the registry.

        Args:
            name (str): The name of the exporter.
            exporter_factory (Callable[[], Awaitable[AbstractExporter]]):
                Factory function that creates a new exporter instance.

        Raises:
            ValueError: If an exporter with the given name already exists.
        """
        async with self._lock:
            if name in self._exporter_factories:
                raise ValueError(f"Exporter with name '{name}' already exists")
            self._exporter_factories[name] = exporter_factory
            logger.info("Added exporter factory '%s'", name)

    async def remove(self, name: str) -> None:
        """
        Remove an exporter factory from the registry by name.

        Args:
            name (str): The name of the exporter to remove.

        Raises:
            ValueError: If no exporter with the given name exists.
        """
        async with self._lock:
            if name not in self._exporter_factories:
                raise ValueError(f"No exporter found with name '{name}'")
            self._exporter_factories.pop(name)
            logger.info("Removed exporter factory '%s'", name)

    async def get(self, name: str) -> AbstractExporter | None:
        """
        Create and retrieve a new exporter instance by name.

        Args:
            name (str): The name of the exporter to retrieve.

        Returns:
            AbstractExporter | None: A new exporter instance if found, else None.
        """
        async with self._lock:
            factory = self._exporter_factories.get(name)
            if factory is None:
                return None
            return await factory()

    async def get_all(self) -> dict[str, AbstractExporter]:
        """
        Create and retrieve new instances of all registered exporters.

        Returns:
            dict[str, AbstractExporter]: A dictionary mapping exporter names to new exporter instances.
        """
        async with self._lock:
            return {name: await factory() for name, factory in self._exporter_factories.items()}
