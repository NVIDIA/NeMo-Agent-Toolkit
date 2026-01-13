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
#

import json
from collections.abc import AsyncGenerator

from nat.data_models.api_server import ResponseIntermediateStep
from nat.front_ends.fastapi import response_helpers


async def test_generate_streaming_response_full_as_str_includes_trace(monkeypatch):
    steps = [
        ResponseIntermediateStep(id="step-1", name="one", payload="{}"),
        ResponseIntermediateStep(id="step-2", name="two", payload="{}"),
    ]

    async def _fake_generate_streaming_response_full(*_, trace_collector=None, **__) -> AsyncGenerator:
        for step in steps:
            if trace_collector is not None:
                trace_collector.append(step)
            yield step

    monkeypatch.setattr(response_helpers, "generate_streaming_response_full", _fake_generate_streaming_response_full)

    chunks = [
        chunk async for chunk in response_helpers.generate_streaming_response_full_as_str(
            None, session=None, streaming=True, include_trace=True)
    ]

    assert chunks, "Expected at least one streamed chunk"
    summary = chunks[-1]
    summary_payload = summary.replace("data: ", "").strip()
    summary_data = json.loads(summary_payload)
    assert summary_data["_trace"]["events"][0]["id"] == "step-1"
    assert len(summary_data["_trace"]["events"]) == 2
