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

import importlib
import logging
from typing import Any

logger = logging.getLogger(__name__)

IS_OTEL_AVAILABLE = None
IS_PHOENIX_AVAILABLE = None


class OptionalImportError(Exception):
    """Raised when an optional import fails."""

    def __init__(self, module_name: str):
        super().__init__(f"Optional dependency '{module_name}' not found. "
                         "If you want to use this feature, please install it with: uv pip install -e '.[telemetry]'")


def optional_import(module_name: str) -> Any:
    """Attempt to import a module, raising OptionalImportError if it fails."""
    try:
        return importlib.import_module(module_name)
    except ImportError as e:
        raise OptionalImportError(module_name) from e


def try_import_opentelemetry() -> Any:
    """Get the opentelemetry module if available."""
    global IS_OTEL_AVAILABLE
    try:
        module = optional_import("opentelemetry")
        IS_OTEL_AVAILABLE = True
        return module
    except OptionalImportError as e:
        if IS_OTEL_AVAILABLE is None:
            # Show the warning only once
            logger.warning("OpenTelemetry not available: %s", e)
            IS_OTEL_AVAILABLE = False
        raise OptionalImportError("opentelemetry") from e


def try_import_phoenix() -> Any:
    """Get the phoenix module if available."""
    global IS_PHOENIX_AVAILABLE
    try:
        module = optional_import("phoenix")
        IS_PHOENIX_AVAILABLE = True
        return module
    except OptionalImportError as e:
        if IS_PHOENIX_AVAILABLE is None:
            # Show the warning only once
            logger.warning("Phoenix not available: %s", e)
            IS_PHOENIX_AVAILABLE = False
        raise OptionalImportError("phoenix") from e


# Dummy OpenTelemetry classes for when the package is not available
class DummySpan:
    """Dummy span class that does nothing when OpenTelemetry is not available."""

    def __init__(self, *args, **kwargs):
        pass

    def end(self, *args, **kwargs):
        pass

    def set_attribute(self, *args, **kwargs):
        pass


class DummyTracer:
    """Dummy tracer class that returns dummy spans."""

    def start_span(self, *args, **kwargs):
        return DummySpan()


class DummyTracerProvider:
    """Dummy tracer provider that returns dummy tracers."""

    @staticmethod
    def get_tracer(*args, **kwargs):
        return DummyTracer()

    @staticmethod
    def add_span_processor(*args, **kwargs):
        pass


class DummyTrace:
    """Dummy trace module that returns dummy tracer providers."""

    @staticmethod
    def get_tracer_provider():
        return DummyTracerProvider()

    @staticmethod
    def set_tracer_provider(*args, **kwargs):
        pass

    @staticmethod
    def get_tracer(*args, **kwargs):
        return DummyTracer()


class DummySpanExporter:
    """Dummy span exporter that does nothing."""

    @staticmethod
    def export(*args, **kwargs):
        pass

    @staticmethod
    def shutdown(*args, **kwargs):
        pass


class DummyBatchSpanProcessor:
    """Dummy implementation of BatchSpanProcessor for when OpenTelemetry is not available."""

    def __init__(self, *args, **kwargs):
        pass

    @staticmethod
    def shutdown(*args, **kwargs):
        pass


# Dummy functions for when OpenTelemetry is not available
def dummy_set_span_in_context(*args, **kwargs) -> None:
    """Dummy function that does nothing."""
    return None
