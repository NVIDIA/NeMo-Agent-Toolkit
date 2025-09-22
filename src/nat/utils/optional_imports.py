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
from types import ModuleType

logger = logging.getLogger(__name__)


class OptionalImportError(Exception):
    """Raised when an optional import fails."""

    def __init__(self, module_name: str, additional_message: str = ""):
        super().__init__(f"Optional dependency '{module_name}' is not installed. {additional_message}")


class TelemetryOptionalImportError(OptionalImportError):
    """Raised when an optional import of telemetry dependencies fails."""

    def __init__(self, module_name: str):
        super().__init__(
            module_name,
            "But the configuration file contains tracing exporters. "
            "If you want to use this feature, please install it with: uv pip install -e '.[telemetry]'",
        )


def optional_import(module_name: str) -> ModuleType:
    """Attempt to import a module, raising OptionalImportError if it fails."""
    try:
        return importlib.import_module(module_name)
    except ImportError as e:
        raise OptionalImportError(module_name) from e


def telemetry_optional_import(module_name: str) -> ModuleType:
    """Attempt to import a module, raising TelemetryOptionalImportError if it fails."""
    try:
        return importlib.import_module(module_name)
    except ImportError as e:
        raise TelemetryOptionalImportError(module_name) from e


def try_import_opentelemetry() -> ModuleType:
    """Get the opentelemetry module if available."""
    return telemetry_optional_import("opentelemetry")


def try_import_phoenix() -> ModuleType:
    """Get the phoenix module if available."""
    return telemetry_optional_import("phoenix")


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
