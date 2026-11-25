# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""The non-interactive HTTP routes must hand the HTTP auth flow handler to the session."""

from contextlib import asynccontextmanager
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest

from nat.data_models.api_server import ChatRequest
from nat.front_ends.fastapi.routes import generate
from nat.front_ends.fastapi.routes import v1_chat_completions


def _make_session_manager(calls: list[dict]) -> MagicMock:

    @asynccontextmanager
    async def _session(**kwargs):
        calls.append(kwargs)
        yield MagicMock()

    session_manager = MagicMock()
    session_manager.session = _session
    return session_manager


def _make_worker() -> MagicMock:
    worker = MagicMock()
    worker.get_step_adaptor.return_value = MagicMock()
    return worker


@pytest.mark.parametrize("stream", [False, True])
async def test_v1_chat_completions_passes_auth_callback(stream: bool):
    calls: list[dict] = []
    worker = _make_worker()
    handler = v1_chat_completions.post_openai_api_compatible_endpoint(worker=worker,
                                                                      session_manager=_make_session_manager(calls),
                                                                      enable_interactive=False)
    payload = ChatRequest.from_string("hi")
    payload.stream = stream

    with patch.object(v1_chat_completions, "generate_single_response", return_value={}), \
            patch.object(v1_chat_completions, "generate_streaming_response_as_str", return_value=iter(())), \
            patch.object(v1_chat_completions, "add_context_headers_to_response"):
        await handler(MagicMock(), MagicMock(), payload)

    assert len(calls) == 1
    assert calls[0]["user_authentication_callback"] is worker._http_flow_handler.authenticate


@pytest.mark.parametrize("factory, patched",
                         [(generate.post_streaming_raw_endpoint, "generate_streaming_response_full_as_str"),
                          (generate.post_streaming_atif_endpoint, "generate_streaming_response_atif_as_str")])
async def test_generate_stream_routes_pass_auth_callback(factory, patched: str):
    calls: list[dict] = []
    worker = _make_worker()
    handler = factory(worker=worker,
                      session_manager=_make_session_manager(calls),
                      request_type=dict,
                      enable_interactive=False,
                      streaming=True,
                      result_type=None,
                      output_type=None)

    with patch.object(generate, patched, return_value=iter(())):
        await handler(MagicMock(), {"input": "x"})

    assert len(calls) == 1
    assert calls[0]["user_authentication_callback"] is worker._http_flow_handler.authenticate
