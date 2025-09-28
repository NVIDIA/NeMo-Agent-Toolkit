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

import pytest
from starlette.requests import Request

from nat.builder.context import ContextState
from nat.runtime.session import SessionManager


class _DummyWorkflow:
    config = None


@pytest.mark.parametrize("header_case", [
    "traceparent",
    "workflow-trace-id",
])
@pytest.mark.asyncio
async def test_session_sets_trace_id_from_headers(header_case: str):
    trace_id_hex = "a" * 32
    span_id_hex = "b" * 16

    if header_case == "traceparent":
        headers = [
            (b"traceparent", f"00-{trace_id_hex}-{span_id_hex}-01".encode("utf-8")),
        ]
    else:
        headers = [
            (b"workflow-trace-id", trace_id_hex.encode("utf-8")),
        ]

    scope = {
        "type": "http",
        "method": "GET",
        "path": "/",
        "headers": headers,
        "client": ("127.0.0.1", 1234),
        "scheme": "http",
        "server": ("testserver", 80),
        "query_string": b"",
    }
    request = Request(scope)

    ctx_state = ContextState.get()
    prev = ctx_state.workflow_trace_id.set(None)
    try:
        sm = SessionManager(workflow=_DummyWorkflow(), max_concurrency=0)
        sm.set_metadata_from_http_request(request)
        assert ctx_state.workflow_trace_id.get() == int(trace_id_hex, 16)
    finally:
        ctx_state.workflow_trace_id.reset(prev)
