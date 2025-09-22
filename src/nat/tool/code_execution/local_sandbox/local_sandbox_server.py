# Copyright (c) 2024-2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import contextlib
import logging
import multiprocessing
import os
import resource
from enum import Enum
from io import StringIO

from flask import Flask
from flask import Request
from flask import Response
from flask import request
from pydantic import BaseModel
from pydantic import Field

app = Flask(__name__)
logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)


class CodeExecutionStatus(str, Enum):
    """
    Status of code execution.
    """
    COMPLETED = "completed"
    ERROR = "error"
    TIMEOUT = "timeout"


class CodeExecutionResult(BaseModel):
    """
    Result of code execution.
    """
    process_status: CodeExecutionStatus = Field(default=CodeExecutionStatus.COMPLETED,
                                                description="Status of the process")
    stdout: str = Field(description="Standard output of the process")
    stderr: str = Field(description="Standard error of the process")


class CodeExecutionResponse(Response):
    """
    Response class that returns a JSON response with the given status code and result.
    """

    def __init__(self, status_code: int, result: CodeExecutionResult):
        super().__init__(status=status_code, mimetype="application/json", response=result.model_dump_json())

    @classmethod
    def with_error(cls, status_code: int, error_message: str) -> 'CodeExecutionResponse':
        return cls(status_code,
                   CodeExecutionResult(process_status=CodeExecutionStatus.ERROR, stdout="", stderr=error_message))


@app.after_request
def add_hsts_header(response):
    response.headers['Strict-Transport-Security'] = 'max-age=31536000; includeSubDomains'
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'SAMEORIGIN'
    response.headers['X-XSS-Protection'] = '1; mode=block'

    return response


def execute_python(generated_code: str, timeout: float) -> CodeExecutionResult:
    """
    Execute Python code in a subprocess.

    Args:
        generated_code: The code to execute
        timeout: The timeout for the execution

    Returns:
        CodeExecutionResult object containing the execution result
    """

    # running in a separate process to ensure any kind of crashes are properly handled
    queue = multiprocessing.Queue()
    process = multiprocessing.Process(target=execute_code_subprocess, args=(generated_code, queue))

    process.start()
    # wait until the process finishes or the timeout expires
    process.join(timeout=timeout)
    if process.exitcode is None:
        process.kill()
        return CodeExecutionResult(process_status=CodeExecutionStatus.TIMEOUT, stdout="", stderr="Timed out\n")

    return queue.get()


# need to memory-limit to avoid common errors of allocating too much
# but this has to be done in a subprocess to not crush server itself
def execute_code_subprocess(generated_code: str, queue):
    """
    Execute code in a subprocess.

    Args:
        generated_code: The code to execute
        queue: The queue to put the result in
    """

    logger.debug("execute_code_subprocess started, PID: %s", os.getpid())

    try:
        limit = 1024 * 1024 * 1024 * 10  # 10gb - somehow with a smaller limit the server dies when numpy is used
        resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
        resource.setrlimit(resource.RLIMIT_DATA, (limit, limit))
    except Exception as e:
        logger.exception("Failed to set resource limits, PID: %s, error: %s", os.getpid(), e)

    stdout_capture = StringIO()
    stderr_capture = StringIO()
    try:
        with contextlib.redirect_stdout(stdout_capture), contextlib.redirect_stderr(stderr_capture):
            exec(generated_code, {})
        logger.debug("execute_code_subprocess finished, PID: %s", os.getpid())
        queue.put(CodeExecutionResult(stdout=stdout_capture.getvalue(), stderr=stderr_capture.getvalue()))
    except Exception as e:
        import traceback
        with contextlib.redirect_stderr(stderr_capture):
            traceback.print_exc()
        logger.debug("execute_code_subprocess failed, PID: %s, error: %s", os.getpid(), e)
        queue.put(
            CodeExecutionResult(process_status=CodeExecutionStatus.ERROR,
                                stdout=stdout_capture.getvalue(),
                                stderr=stderr_capture.getvalue()))


def do_execute(request: Request) -> CodeExecutionResponse:
    """
    Main function to handle execution requests.

    Args:
        request: Request object containing the execution request

    Returns:
        CodeExecutionResponse object containing the execution result
    """
    try:
        # Check if request has JSON data
        if not request.is_json:
            return CodeExecutionResponse.with_error(400, "Request must be JSON")

        # Get JSON data safely
        json_data = request.get_json(silent=True)

        if json_data is None:
            return CodeExecutionResponse.with_error(400, "Invalid JSON data")

        # Check for required fields
        if 'generated_code' not in json_data:
            return CodeExecutionResponse.with_error(400, "Missing required field: generated_code")

        if 'timeout' not in json_data:
            return CodeExecutionResponse.with_error(400, "Missing required field: timeout")

        if 'language' not in json_data:
            return CodeExecutionResponse.with_error(400, "Missing required field: language")

        generated_code: str | None = json_data.get('generated_code', None)
        assert generated_code is not None
        timeout: float | None = json_data.get('timeout', None)
        assert timeout is not None
        language: str | None = json_data.get('language', None)
        assert language is not None

        if language != 'python':
            return CodeExecutionResponse.with_error(400, "Only python execution is supported")

        return CodeExecutionResponse(200, execute_python(generated_code, timeout))

    except Exception as e:
        return CodeExecutionResponse.with_error(500, f"Server error: {str(e)}")


# Main Flask endpoint to handle execution requests
@app.route("/execute", methods=["POST"])
def execute():
    return do_execute(request)


if __name__ == '__main__':
    app.run(port=6000)
