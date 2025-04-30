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
from typing import Optional

logger = logging.getLogger(__name__)


class OptionalImportError(Exception):
    """Raised when an optional import fails."""

    def __init__(self, module_name: str):
        super().__init__(f"Optional dependency '{module_name}' not found. "
                         "Please install it with: uv pip install aiq[telemetry]")


def optional_import(module_name: str) -> Any:
    """Attempt to import a module, raising OptionalImportError if it fails."""
    try:
        return importlib.import_module(module_name)
    except ImportError as e:
        raise OptionalImportError(module_name) from e


def get_opentelemetry() -> Any:
    """Get the opentelemetry module if available."""
    return optional_import("opentelemetry")


def get_opentelemetry_sdk() -> Any:
    """Get the opentelemetry.sdk module if available."""
    return optional_import("opentelemetry.sdk")


def get_phoenix() -> Any:
    """Get the phoenix module if available."""
    return optional_import("phoenix")


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

    def get_tracer(self, *args, **kwargs):
        return DummyTracer()

    def add_span_processor(self, *args, **kwargs):
        pass


class DummyTrace:
    """Dummy trace module that returns dummy tracer providers."""

    def get_tracer_provider(self):
        return DummyTracerProvider()

    def set_tracer_provider(self, *args, **kwargs):
        pass

    def get_tracer(self, *args, **kwargs):
        return DummyTracer()


class DummySpanExporter:
    """Dummy span exporter that does nothing."""

    def export(self, *args, **kwargs):
        pass

    def shutdown(self, *args, **kwargs):
        pass


# Dummy functions for when OpenTelemetry is not available
def dummy_set_span_in_context(*args, **kwargs) -> None:
    """Dummy function that does nothing."""
    return None


# Lazy singleton instances
_dummy_trace: Optional[DummyTrace] = None
_dummy_tracer_provider: Optional[DummyTracerProvider] = None
_dummy_span_exporter: Optional[DummySpanExporter] = None


def get_dummy_trace() -> DummyTrace:
    """Get the singleton dummy trace instance, creating it if needed."""
    global _dummy_trace
    if _dummy_trace is None:
        _dummy_trace = DummyTrace()
    return _dummy_trace


def get_dummy_tracer_provider() -> DummyTracerProvider:
    """Get the singleton dummy tracer provider instance, creating it if needed."""
    global _dummy_tracer_provider
    if _dummy_tracer_provider is None:
        _dummy_tracer_provider = DummyTracerProvider()
    return _dummy_tracer_provider


def get_dummy_span_exporter() -> DummySpanExporter:
    """Get the singleton dummy span exporter instance, creating it if needed."""
    global _dummy_span_exporter
    if _dummy_span_exporter is None:
        _dummy_span_exporter = DummySpanExporter()
    return _dummy_span_exporter
