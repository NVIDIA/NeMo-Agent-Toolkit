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

import typing
import uuid

import pytest
from starlette.requests import Request

from nat.builder.context import ContextState
from nat.builder.workflow import Workflow
from nat.runtime.session import SessionManager


class _DummyWorkflow:
    config = None


# Build parameter sets at import time to keep test bodies simple
_random_trace_hex = uuid.uuid4().hex
_random_workflow_uuid_hex = uuid.uuid4().hex
_random_workflow_uuid_str = str(uuid.uuid4())

TRACE_ID_CASES: list[tuple[list[tuple[bytes, bytes]], int | None]] = [
    # traceparent valid cases
    ([(b"traceparent", f"00-{'a'*32}-{'b'*16}-01".encode())], int("a" * 32, 16)),
    ([(b"traceparent", f"00-{'A'*32}-{'b'*16}-01".encode())], int("A" * 32, 16)),
    ([(b"traceparent", f"00-{_random_trace_hex}-{'b'*16}-01".encode())], int(_random_trace_hex, 16)),
    # workflow-trace-id valid cases (hex and hyphenated)
    ([(b"workflow-trace-id", _random_workflow_uuid_hex.encode())], uuid.UUID(_random_workflow_uuid_hex).int),
    ([(b"workflow-trace-id", _random_workflow_uuid_str.encode())], uuid.UUID(_random_workflow_uuid_str).int),
    # invalid traceparent falls back to workflow-trace-id
    ([
        (b"traceparent", f"00-{'a'*31}-{'b'*16}-01".encode()),
        (b"workflow-trace-id", _random_workflow_uuid_str.encode()),
    ],
     uuid.UUID(_random_workflow_uuid_str).int),
    # invalid both -> None
    ([
        (b"traceparent", f"00-{'g'*32}-{'b'*16}-01".encode()),
        (b"workflow-trace-id", b"z" * 32),
    ], None),
    # prefer traceparent when both valid
    ([
        (b"traceparent", f"00-{'c'*32}-{'d'*16}-01".encode()),
        (b"workflow-trace-id", str(uuid.uuid4()).encode()),
    ],
     int("c" * 32, 16)),
    # zero values
    ([(b"traceparent", f"00-{'0'*32}-{'b'*16}-01".encode())], 0),
    ([(b"workflow-trace-id", ("0" * 32).encode())], 0),
    # malformed span id but valid trace id
    ([(b"traceparent", f"00-{'a'*32}-XYZ-01".encode())], int("a" * 32, 16)),
    # too few parts -> ignore
    ([(b"traceparent", f"00-{'a'*32}".encode())], None),
    # extra parts -> still ok
    ([(b"traceparent", f"00-{'b'*32}-{'c'*16}-01-extra".encode())], int("b" * 32, 16)),
    # negative and overflow workflow-trace-id -> ignore
    ([(b"workflow-trace-id", b"-1")], None),
    ([(b"workflow-trace-id", ("f" * 33).encode())], None),
]


@pytest.mark.parametrize(
    "headers,expected_trace_id",
    TRACE_ID_CASES,
)
@pytest.mark.asyncio
async def test_session_trace_id_from_headers_parameterized(headers: list[tuple[bytes, bytes]],
                                                           expected_trace_id: int | None):
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
    token = ctx_state.workflow_trace_id.set(None)
    try:
        sm = SessionManager(workflow=typing.cast(Workflow, _DummyWorkflow()), max_concurrency=0)
        sm.set_metadata_from_http_request(request)
        assert ctx_state.workflow_trace_id.get() == expected_trace_id
    finally:
        ctx_state.workflow_trace_id.reset(token)


METADATA_CASES: list[tuple[list[tuple[bytes, bytes]], str | None, str | None, str | None]] = [
    ([(b"conversation-id", b"conv-123")], "conv-123", None, None),
    ([(b"user-message-id", b"msg-456")], None, "msg-456", None),
    ([(b"workflow-run-id", b"run-789")], None, None, "run-789"),
    (
        [
            (b"conversation-id", b"conv-123"),
            (b"user-message-id", b"msg-456"),
            (b"workflow-run-id", b"run-789"),
            (b"traceparent", f"00-{'e'*32}-{'f'*16}-01".encode()),
        ],
        "conv-123",
        "msg-456",
        "run-789",
    ),
]


@pytest.mark.parametrize(
    "headers,expected_conv,expected_msg,expected_run",
    METADATA_CASES,
)
@pytest.mark.asyncio
async def test_session_metadata_headers_parameterized(headers: list[tuple[bytes, bytes]],
                                                      expected_conv: str | None,
                                                      expected_msg: str | None,
                                                      expected_run: str | None):
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
    tkn_conv = ctx_state.conversation_id.set(None)
    tkn_msg = ctx_state.user_message_id.set(None)
    tkn_run = ctx_state.workflow_run_id.set(None)
    tkn_trace = ctx_state.workflow_trace_id.set(None)
    try:
        sm = SessionManager(workflow=typing.cast(Workflow, _DummyWorkflow()), max_concurrency=0)
        sm.set_metadata_from_http_request(request)
        assert ctx_state.conversation_id.get() == expected_conv
        assert ctx_state.user_message_id.get() == expected_msg
        assert ctx_state.workflow_run_id.get() == expected_run
    finally:
        ctx_state.conversation_id.reset(tkn_conv)
        ctx_state.user_message_id.reset(tkn_msg)
        ctx_state.workflow_run_id.reset(tkn_run)
        ctx_state.workflow_trace_id.reset(tkn_trace)
