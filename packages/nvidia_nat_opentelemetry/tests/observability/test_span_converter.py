# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import os

import pytest
from openinference.semconv.trace import OpenInferenceSpanKindValues
from openinference.semconv.trace import SpanAttributes as OpenInferenceSpanAttributes

import nat.data_models.span as nat_span_module
from nat.plugins.opentelemetry.span_converter import convert_span_to_otel

_CONVERTER_MODULE = "nat.plugins.opentelemetry.span_converter"


@pytest.fixture
def reloaded_with_span_prefix(request):
    """Reload the NAT span modules with ``NAT_SPAN_PREFIX`` set.

    ``SpanAttributes`` is built at import time, so honoring a custom prefix requires reloading both
    the module that defines the attribute keys and the converter that reads them. The original
    environment and module objects are restored on teardown.
    """
    prefix = request.param

    original_prefix = os.environ.get("NAT_SPAN_PREFIX")
    os.environ["NAT_SPAN_PREFIX"] = prefix
    try:
        span_module = importlib.reload(nat_span_module)
        converter_module = importlib.reload(importlib.import_module(_CONVERTER_MODULE))
        yield span_module, converter_module
    finally:
        if original_prefix is None:
            os.environ.pop("NAT_SPAN_PREFIX", None)
        else:
            os.environ["NAT_SPAN_PREFIX"] = original_prefix
        importlib.reload(nat_span_module)
        importlib.reload(importlib.import_module(_CONVERTER_MODULE))


@pytest.mark.parametrize("reloaded_with_span_prefix", ["nat"], indirect=True)
def test_converted_span_kind_with_default_prefix(reloaded_with_span_prefix):
    """A span tagged with the default ``nat`` prefix keeps its OpenInference span kind.

    The modules are reloaded with ``NAT_SPAN_PREFIX`` pinned to ``nat`` so the assertion does not
    depend on the prefix the test process happens to start with.
    """
    span_module, converter_module = reloaded_with_span_prefix

    nat_span = span_module.Span(name="llm", attributes={"nat.event_type": "LLM_START"})

    otel_span = converter_module.convert_span_to_otel(nat_span)

    assert otel_span.attributes[OpenInferenceSpanAttributes.OPENINFERENCE_SPAN_KIND] == (
        OpenInferenceSpanKindValues.LLM.value)


@pytest.mark.parametrize("reloaded_with_span_prefix", ["nat", "custom", "prod.service"], indirect=True)
def test_converted_span_kind_uses_configured_span_prefix(reloaded_with_span_prefix):
    """The OpenInference span kind is resolved from the configured span prefix.

    ``SpanExporter`` writes the event type under ``f"{NAT_SPAN_PREFIX}.event_type"``, so a prefix
    other than the default must not degrade every span to ``UNKNOWN``.
    """
    span_module, converter_module = reloaded_with_span_prefix

    event_type_key = span_module.SpanAttributes.NAT_EVENT_TYPE.value
    nat_span = span_module.Span(name="llm", attributes={event_type_key: "LLM_START"})

    otel_span = converter_module.convert_span_to_otel(nat_span)

    assert otel_span.attributes[OpenInferenceSpanAttributes.OPENINFERENCE_SPAN_KIND] == (
        OpenInferenceSpanKindValues.LLM.value)


def test_converted_span_kind_without_event_type_is_unknown():
    """A span carrying no event type attribute is reported as ``UNKNOWN``."""
    nat_span = nat_span_module.Span(name="opaque", attributes={})

    otel_span = convert_span_to_otel(nat_span)

    assert otel_span.attributes[OpenInferenceSpanAttributes.OPENINFERENCE_SPAN_KIND] == (
        OpenInferenceSpanKindValues.UNKNOWN.value)
