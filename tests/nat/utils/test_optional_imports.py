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

from nat.utils.optional_imports import DummyBatchSpanProcessor
from nat.utils.optional_imports import DummySpan
from nat.utils.optional_imports import DummySpanExporter
from nat.utils.optional_imports import DummyTrace
from nat.utils.optional_imports import DummyTracerProvider
from nat.utils.optional_imports import OptionalImportError
from nat.utils.optional_imports import TelemetryOptionalImportError
from nat.utils.optional_imports import optional_import
from nat.utils.optional_imports import telemetry_optional_import


def test_optional_import_success():
    assert optional_import("math").sqrt(4) == 2


def test_optional_import_failure():
    with pytest.raises(OptionalImportError):
        optional_import("nonexistent___module___xyz")


def test_telemetry_optional_import_failure_has_guidance():
    with pytest.raises(TelemetryOptionalImportError) as ei:
        telemetry_optional_import("not_real_otel_mod")
    assert "Optional dependency" in str(ei.value)
    assert "telemetry" in str(ei.value).lower()


def test_dummy_tracer_stack():
    tracer = DummyTracerProvider.get_tracer()
    span = tracer.start_span("op")
    assert isinstance(span, DummySpan)
    span.set_attribute("k", "v")
    span.end()
    DummyBatchSpanProcessor().shutdown()
    DummySpanExporter.export()
    DummySpanExporter.shutdown()
    assert DummyTrace.get_tracer_provider() is not None
    DummyTrace.set_tracer_provider(None)
    assert DummyTrace.get_tracer("name") is not None
