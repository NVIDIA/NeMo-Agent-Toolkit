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
import os
import weakref
from typing import Any

import ragaai_catalyst
from ragaai_catalyst.tracers.exporters import DynamicTraceExporter

from aiq.plugins.opentelemetry.otel_span import OtelSpan

logger = logging.getLogger(__name__)


class ExporterSettings:
    """Settings for a specific exporter instance."""

    def __init__(self, disable_local_file: bool, local_file_path: str | None):
        self.disable_local_file = disable_local_file
        self.local_file_path = local_file_path

    def __repr__(self):
        return f"ExporterSettings(disable_local_file={self.disable_local_file}, local_file_path={self.local_file_path})"


class RagaAICatalystMixin:
    """Mixin for RagaAI Catalyst exporters.

    This mixin provides RagaAI Catalyst-specific functionality for OpenTelemetry span exporters.
    It handles RagaAI Catalyst project and dataset configuration and uses the DynamicTraceExporter
    from the ragaai_catalyst.tracers.exporters module.

    Key Features:
    - RagaAI Catalyst authentication with access key and secret key
    - Project and dataset scoping for trace organization
    - Integration with RagaAI Catalyst's DynamicTraceExporter for telemetry transmission
    - Automatic initialization of RagaAI Catalyst client
    - Per-instance local file control for rag_agent_traces.json management

    This mixin is designed to be used with OtelSpanExporter as a base class:

    Example:
        class MyCatalystExporter(OtelSpanExporter, RagaAICatalystMixin):
            def __init__(self, base_url, access_key, secret_key, project, dataset, **kwargs):
                super().__init__(base_url=base_url, access_key=access_key,
                               secret_key=secret_key, project=project, dataset=dataset, **kwargs)
    """

    # Class variables (shared across all instances)
    _exporter_settings_registry: weakref.WeakKeyDictionary[Any, ExporterSettings] = weakref.WeakKeyDictionary()
    _hooks_applied = False

    def __init__(self,
                 *args,
                 base_url: str,
                 access_key: str,
                 secret_key: str,
                 project: str,
                 dataset: str,
                 disable_local_file: bool = False,
                 local_file_path: str | None = None,
                 **kwargs):
        """Initialize the RagaAI Catalyst exporter.

        Args:
            base_url: RagaAI Catalyst base URL.
            access_key: RagaAI Catalyst access key.
            secret_key: RagaAI Catalyst secret key.
            project: RagaAI Catalyst project name.
            dataset: RagaAI Catalyst dataset name.
            disable_local_file: Disable creation of local rag_agent_traces.json file.
            local_file_path: Custom path to save local trace files instead of current directory.
        """
        logger.info("RagaAICatalystMixin initialized with disable_local_file=%s, local_file_path=%s",
                    disable_local_file,
                    local_file_path)

        ragaai_catalyst.RagaAICatalyst(access_key=access_key, secret_key=secret_key, base_url=base_url)

        # Store settings for this instance
        self._exporter_settings = ExporterSettings(disable_local_file, local_file_path)

        # Apply hooks if not already applied
        self._ensure_hooks_applied()

        # Create the DynamicTraceExporter (this will trigger our hook)
        self._exporter = DynamicTraceExporter(project, dataset, base_url, "agentic/nemo-framework")

        # Register the internal RAGATraceExporter with our settings
        self._register_exporter_settings()

        super().__init__(*args, **kwargs)

    def _ensure_hooks_applied(self):
        """Ensure the hooks are applied exactly once."""
        if not self.__class__._hooks_applied:
            self._apply_hooks()
            self.__class__._hooks_applied = True
            logger.info("Applied RagaAI Catalyst hooks for per-instance local file control")

    def _apply_hooks(self):
        """Apply the monkey patches for per-instance control."""
        self._hook_dynamic_trace_exporter()
        self._hook_raga_trace_exporter()

    def _hook_dynamic_trace_exporter(self):
        """Hook into DynamicTraceExporter to capture the internal RAGATraceExporter."""
        try:
            # We don't actually need to hook DynamicTraceExporter since we can access
            # the internal exporter after creation via self._exporter._exporter
            pass
        except Exception as e:
            logger.error("Failed to hook DynamicTraceExporter: %s", e)

    def _hook_raga_trace_exporter(self):
        """Hook into RAGATraceExporter.prepare_trace to use per-instance settings."""
        try:
            # Import the module we need to patch
            import ragaai_catalyst.tracers.exporters.ragaai_trace_exporter as raga_exporter
        except ImportError:
            logger.warning("ragaai_catalyst package not found - local file control patch not applied")
            return

        try:
            # Check if patch is already applied to avoid double patching
            if hasattr(raga_exporter.RAGATraceExporter.prepare_trace, '_aiq_patched'):
                logger.debug("RagaAI local file control patch already applied")
                return

            # Save the original method
            original_prepare_trace = raga_exporter.RAGATraceExporter.prepare_trace

            def patched_prepare_trace(self, spans, trace_id):
                """
                Patched version that calls the original method but controls local file creation per instance.
                """
                logger.debug("Patched prepare_trace called for trace_id=%s", trace_id)

                # Call the original method (which creates the file)
                result = original_prepare_trace(self, spans, trace_id)

                # Look up settings for this specific exporter instance
                settings = RagaAICatalystMixin._exporter_settings_registry.get(self, None)
                logger.debug("Found settings for exporter %s: %s", id(self), settings)

                if settings:
                    local_file_path = os.path.join(os.getcwd(), 'rag_agent_traces.json')
                    logger.debug("Checking local file: %s (exists=%s)",
                                 local_file_path,
                                 os.path.exists(local_file_path))

                    if settings.disable_local_file:
                        logger.debug("Attempting to remove local file...")
                        # Remove the hardcoded file if it exists
                        try:
                            if os.path.exists(local_file_path):
                                os.remove(local_file_path)
                                logger.info("Removed local trace file: %s", local_file_path)
                            else:
                                logger.debug("Local trace file does not exist: %s", local_file_path)
                        except (OSError, IOError) as e:
                            logger.warning("Could not remove local trace file: %s", e)

                    elif settings.local_file_path:
                        logger.debug("Attempting to move local file to custom path...")
                        # Move file to custom location
                        try:
                            if os.path.exists(local_file_path):
                                os.makedirs(settings.local_file_path, exist_ok=True)
                                new_path = os.path.join(settings.local_file_path, f'trace_{trace_id}.json')
                                os.rename(local_file_path, new_path)
                                logger.info("Moved trace file to: %s", new_path)
                            else:
                                logger.debug("Local trace file does not exist to move: %s", local_file_path)
                        except (OSError, IOError) as e:
                            logger.warning("Could not move trace file: %s", e)
                else:
                    logger.debug("No settings found for exporter %s, using default behavior", id(self))

                return result

            # Apply the patch and mark it as applied
            raga_exporter.RAGATraceExporter.prepare_trace = patched_prepare_trace
            raga_exporter.RAGATraceExporter.prepare_trace._aiq_patched = True

            logger.info("Applied RagaAI local file control patch for per-instance control")

        except AttributeError as e:
            logger.error("Failed to patch ragaai_catalyst - method not found: %s", e)
        except Exception as e:
            logger.error("Failed to patch ragaai_catalyst: %s", e)

    def _register_exporter_settings(self):
        """Register the settings for the internal RAGATraceExporter."""
        try:
            # Access the internal RAGATraceExporter from DynamicTraceExporter
            internal_exporter = getattr(self._exporter, '_exporter', None)
            if internal_exporter:
                self.__class__._exporter_settings_registry[internal_exporter] = self._exporter_settings
                logger.debug("Registered settings for exporter %s: %s", id(internal_exporter), self._exporter_settings)
            else:
                logger.warning("Could not access internal RAGATraceExporter from DynamicTraceExporter")
        except Exception as e:
            logger.error("Failed to register exporter settings: %s", e)

    @classmethod
    def get_registry_size(cls) -> int:
        """Get the current size of the exporter settings registry.

        This is useful for testing and debugging.

        Returns:
            int: Number of registered exporters
        """
        return len(cls._exporter_settings_registry)

    @classmethod
    def clear_registry(cls) -> None:
        """Clear the exporter settings registry.

        This is useful for testing cleanup.
        """
        cls._exporter_settings_registry.clear()
        logger.debug("Cleared exporter settings registry")

    async def export_otel_spans(self, spans: list[OtelSpan]) -> None:
        """Export a list of OtelSpans using the RagaAI Catalyst exporter.

        Args:
            spans (list[OtelSpan]): The list of spans to export.

        Raises:
            Exception: If there's an error during span export (logged but not re-raised).
        """
        try:
            # Run the blocking export operation in a thread pool to make it non-blocking
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, lambda: self._exporter.export(spans))  # type: ignore[arg-type]
        except Exception as e:
            logger.error("Error exporting spans: %s", e, exc_info=True)
