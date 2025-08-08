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

import json
import uuid
from pathlib import Path

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient
from aiohttp.test_utils import TestServer

from nat.data_models.api_server import ResponseIntermediateStep
from nat.eval.config import EvaluationRunConfig
from nat.eval.remote_workflow import EvaluationRemoteWorkflowHandler


@pytest.fixture
def rag_streamed_intermediate_payloads(rag_intermediate_steps) -> list[str]:
    """
    Returns a list of `intermediate_data:` lines as they would be streamed from the server.
    """
    streamed_lines = []

    # Use the first list of steps
    steps1, steps2 = rag_intermediate_steps
    for step in steps1:
        wrapped = ResponseIntermediateStep(id=str(uuid.uuid4()),
                                           name=step.name or "",
                                           parent_id=step.parent_id,
                                           type=step.event_type,
                                           payload=step.payload.model_dump_json())
        streamed_lines.append(f"intermediate_data: {wrapped.model_dump_json()}\n")

    return streamed_lines


@pytest.fixture
def stream_response_app(rag_eval_input, rag_streamed_intermediate_payloads):
    """
    Returns an aiohttp app with a /generate/full route that simulates streaming:
    - One final output (data line)
    - Several intermediate steps (intermediate_data lines)
    """
    final_output = rag_eval_input.eval_input_items[0].output_obj

    async def stream_response(request):
        resp = web.StreamResponse(status=200, reason="OK", headers={"Content-Type": "text/event-stream"})
        await resp.prepare(request)

        # Final workflow output
        data_line = f"data: {json.dumps({'value': final_output})}\n\n"
        await resp.write(data_line.encode("utf-8"))

        # Intermediate steps
        for line in rag_streamed_intermediate_payloads:
            await resp.write(f"{line}\n".encode("utf-8"))

        await resp.write_eof()
        return resp

    app = web.Application()
    app.router.add_post("/generate/full", stream_response)
    return app


async def test_run_workflow_remote_single_success(stream_response_app, rag_eval_input, rag_intermediate_steps):
    """
    Test parsing of streamed intermediate steps and final output.
    """
    item = rag_eval_input.eval_input_items[0]

    server = TestServer(stream_response_app)
    await server.start_server()
    server_url = str(server.make_url("")).rstrip("/")

    # Run evaluation with the test server.
    # Endpoint and endpoint_timeout are the only fields that are used
    eval_run_config = EvaluationRunConfig(endpoint=server_url,
                                          endpoint_timeout=5,
                                          config_file=Path(__file__),
                                          dataset=None,
                                          result_json_path="",
                                          skip_workflow=False,
                                          skip_completed_entries=False,
                                          reps=1)

    client = TestClient(server)
    await client.start_server()

    handler = EvaluationRemoteWorkflowHandler(config=eval_run_config, max_concurrency=2)

    async with client.session as session:
        await handler.run_workflow_remote_single(session, item)

    await client.close()
    await server.close()

    # Check that the output and trajectory are as expected
    assert item.output_obj == rag_eval_input.eval_input_items[0].output_obj
    # Check that the trajectory contains the expected number of intermediate steps
    steps1, steps2 = rag_intermediate_steps
    assert len(item.trajectory) == len(steps1)


async def test_run_workflow_remote_single_with_invalid_intermediate_data(rag_eval_input):
    """
    Test that malformed intermediate_data lines are logged and skipped gracefully.
    """
    item = rag_eval_input.eval_input_items[0]
    final_output = item.output_obj

    async def stream_response(request):
        resp = web.StreamResponse(status=200, headers={"Content-Type": "text/event-stream"})
        await resp.prepare(request)

        # Valid final output
        await resp.write(f"data: {json.dumps({'value': final_output})}\n\n".encode("utf-8"))

        # Malformed intermediate step (invalid JSON)
        await resp.write(b"intermediate_data: {not a valid json string}\n")

        # Malformed intermediate step (payload is not a stringified JSON)
        bad_payload = {"id": "xyz", "payload": {"event_type": "TOOL_START"}}
        await resp.write(f"intermediate_data: {json.dumps(bad_payload)}\n".encode("utf-8"))

        await resp.write_eof()
        return resp

    app = web.Application()
    app.router.add_post("/generate/full", stream_response)
    server = TestServer(app)
    await server.start_server()

    client = TestClient(server)
    await client.start_server()

    eval_run_config = EvaluationRunConfig(endpoint=str(server.make_url("")).rstrip("/"),
                                          endpoint_timeout=5,
                                          config_file=Path(__file__),
                                          dataset=None,
                                          result_json_path="",
                                          skip_workflow=False,
                                          skip_completed_entries=False,
                                          reps=1)

    handler = EvaluationRemoteWorkflowHandler(config=eval_run_config, max_concurrency=2)

    async with client.session as session:
        await handler.run_workflow_remote_single(session, item)

    await client.close()
    await server.close()

    # Should still receive the final output
    assert item.output_obj == final_output

    # Malformed intermediate steps should be skipped, so trajectory should be empty
    assert item.trajectory == []


async def test_run_workflow_remote_single_with_connection_error(rag_eval_input):
    """
    Test that aiohttp connection errors are handled gracefully.
    """
    item = rag_eval_input.eval_input_items[0]

    # This is an intentionally invalid endpoint that will fail to connect
    eval_run_config = EvaluationRunConfig(
        endpoint="http://127.0.0.1:9999",  # Assuming this port is unused
        endpoint_timeout=1,  # Keep timeout short
        config_file=Path(__file__),
        dataset=None,
        result_json_path="",
        skip_workflow=False,
        skip_completed_entries=False,
        reps=1)

    handler = EvaluationRemoteWorkflowHandler(config=eval_run_config, max_concurrency=2)

    import aiohttp
    timeout = aiohttp.ClientTimeout(total=eval_run_config.endpoint_timeout)

    async with aiohttp.ClientSession(timeout=timeout) as session:
        await handler.run_workflow_remote_single(session, item)

    # Should fail gracefully: no output, no trajectory
    assert item.output_obj is None
    assert item.trajectory == []
