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

from unittest.mock import MagicMock
from unittest.mock import patch

import pytest
import weave
from weave.utils import sanitize

from nat.plugins.weave.register import WeaveTelemetryExporter
from nat.plugins.weave.register import weave_telemetry_exporter


@pytest.fixture
def redact_key():
    key = "custom_secret"
    yield key
    sanitize.remove_redact_key(key)


@pytest.mark.parametrize("redact_pii", [False, True])
async def test_weave_telemetry_exporter_registers_custom_redact_keys(redact_key, redact_pii):
    config = WeaveTelemetryExporter(project="p", redact_keys=[redact_key], redact_pii=redact_pii)
    payload = {redact_key: "v", "other": "x"}

    assert weave.trace.weave_client.redact_sensitive_keys(payload)[redact_key] == "v"

    with patch("weave.init"), patch("nat.plugins.weave.weave_exporter.WeaveExporter"):
        async with weave_telemetry_exporter(config, MagicMock()):
            redacted = weave.trace.weave_client.redact_sensitive_keys(payload)

    assert redacted == {redact_key: "REDACTED", "other": "x"}


async def test_weave_telemetry_exporter_without_redact_keys_adds_no_key(redact_key):
    config = WeaveTelemetryExporter(project="p")
    payload = {redact_key: "v", "other": "x"}

    assert weave.trace.weave_client.redact_sensitive_keys(payload)[redact_key] == "v"

    with patch("weave.init"), patch("nat.plugins.weave.weave_exporter.WeaveExporter"):
        async with weave_telemetry_exporter(config, MagicMock()):
            redacted = weave.trace.weave_client.redact_sensitive_keys(payload)

    assert redacted == payload
