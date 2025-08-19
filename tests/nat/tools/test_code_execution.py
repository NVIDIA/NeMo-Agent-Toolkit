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

import logging
from urllib.parse import urljoin

import pytest
import requests
from pytest_httpserver import HTTPServer

from nat.tool.code_execution import code_sandbox
from nat.tool.code_execution.local_sandbox.local_sandbox_server import do_execute

logger = logging.getLogger(__name__)


def test_client_init(uri: str = "http://localhost:6000"):
    sandbox = code_sandbox.get_sandbox("local", uri=uri)
    assert isinstance(sandbox, code_sandbox.LocalSandbox)
    assert sandbox.url == str(urljoin(uri, "execute"))

    uri = uri + "/"
    sandbox = code_sandbox.get_sandbox("local", uri=uri)
    assert isinstance(sandbox, code_sandbox.LocalSandbox)
    assert sandbox.url == str(urljoin(uri, "execute"))


async def test_handle_response(httpserver: HTTPServer):
    client = code_sandbox.get_sandbox("local", uri=httpserver.url_for("/execute"))
    httpserver.expect_request(
        "/execute",
        method="POST",
    ).respond_with_json({
        "process_status": "completed", "stdout": "Hello World", "stderr": ""
    })

    resp = await client.execute_code(generated_code='print("Hello World")')
    assert isinstance(resp, dict)
    assert resp == {"process_status": "completed", "stdout": "Hello World", "stderr": ""}


async def test_bad_response(httpserver: HTTPServer):
    client = code_sandbox.get_sandbox("local", uri="http://localhost:9999")

    # Test that connection error is raised when the service is unavailable
    with pytest.raises(requests.exceptions.ConnectionError):
        _ = await client.execute_code(generated_code='print("Hello World")')

    # Test for JSON parsing error
    client = code_sandbox.get_sandbox("local", uri=httpserver.url_for("/execute"))
    httpserver.expect_request(
        "/execute",
        method="POST",
    ).respond_with_data("""
                                      "process_status": "completed",
                                      "stdout": "Hello World",
                                      "stderr",  "",
                                   }""")

    resp = await client.execute_code(generated_code='print("Hello World")')
    assert resp.get("process_status") == "error"
    assert resp.get("stdout") == ""
    assert resp.get("stderr").startswith("Unknown error")


async def test_code_gen(httpserver: HTTPServer):

    client = code_sandbox.get_sandbox("local", uri=httpserver.url_for("/execute"))
    httpserver.expect_request("/execute", method="POST").respond_with_handler(do_execute)

    # Execute simple code
    resp = await client.execute_code(generated_code='print("Hello World")')
    assert resp.get("process_status") == "completed"
    assert resp.get("stdout").rstrip() == "Hello World"
    assert resp.get("stderr") == ""

    # Check Timeout
    resp = await client.execute_code(generated_code="import time; time.sleep(5)", timeout_seconds=2)
    assert resp.get("process_status") == "timeout"
    assert resp.get("stdout") == ""
    assert resp.get("stderr").rstrip() == "Timed out"

    # Check Exception
    resp = await client.execute_code(generated_code="print(1/0)")
    assert resp.get("process_status") == "error"
    assert resp.get("stdout") == ""
    assert resp.get("stderr").startswith("Traceback")

    # Check invalid code
    resp = await client.execute_code(generated_code="124ijfmpoeqfmpew')")
    assert resp.get("process_status") == "error"
    assert resp.get("stdout") == ""
    assert resp.get("stderr").startswith("Traceback")

    # Check handle code block
    resp = await client.execute_code(generated_code="""
```
import json


print(5+5)
```
""")
    assert resp.get("process_status") == "completed"
    assert resp.get("stdout").rstrip() == "10"
    assert resp.get("stderr") == ""
