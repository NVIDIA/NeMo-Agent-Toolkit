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

import abc
import json
import logging
import textwrap
from typing import Any
from urllib.parse import urljoin

import requests
import requests.adapters
from pydantic import HttpUrl

from nat.utils.type_utils import override

logger = logging.getLogger(__file__)


class Sandbox(abc.ABC):
    """Code execution sandbox.

    Args:
        host: Optional[str] = '127.0.0.1' - Host of the sandbox server.
            Can also be specified through NEMO_SKILLS_SANDBOX_HOST env var.
        port: Optional[str] = '5000' - Port of the sandbox server.
            Can also be specified through NEMO_SKILLS_SANDBOX_PORT env var.
        ssh_server: Optional[str] = None - SSH server for tunneling requests.
            Useful if server is running on slurm cluster to which there is an ssh access.
            Can also be specified through NEMO_SKILLS_SSH_SERVER env var.
        ssh_key_path: Optional[str] = None - Path to the ssh key for tunneling.
            Can also be specified through NEMO_SKILLS_SSH_KEY_PATH env var.
    """

    def __init__(
        self,
        *,
        uri: HttpUrl,
    ):
        self.url: str = self._get_execute_url(uri)
        session = requests.Session()
        adapter = requests.adapters.HTTPAdapter(pool_maxsize=1500, pool_connections=1500, max_retries=3)
        session.mount('http://', adapter)
        session.mount('https://', adapter)
        self.http_session: requests.Session = session

    def _send_request(self, request: dict[str, Any], timeout_seconds: float) -> dict[str, str]:
        output = self.http_session.post(
            url=self.url,
            data=json.dumps(request),
            timeout=timeout_seconds,
            headers={"Content-Type": "application/json"},
        )
        # retrying 502 errors
        if output.status_code == 502:
            raise requests.exceptions.Timeout

        return self._parse_request_output(output)

    @abc.abstractmethod
    def _parse_request_output(self, output: requests.Response) -> dict[str, str]:
        pass

    @abc.abstractmethod
    def _get_execute_url(self, uri: HttpUrl) -> str:
        pass

    @abc.abstractmethod
    def _prepare_request(self, generated_code: str, timeout_seconds: float) -> dict[str, Any]:
        pass

    async def execute_code(
        self,
        generated_code: str,
        timeout_seconds: float = 10.0,
        language: str = "python",
        max_output_characters: int = 1000,
    ) -> dict[str, str]:

        if language != "python":
            raise ValueError(f"Language {language} not supported")

        generated_code = generated_code.strip().strip("`")
        # Use json.dumps to properly escape the generated_code instead of repr()
        escaped_code = json.dumps(generated_code)
        code_to_execute = textwrap.dedent(f"""
            import traceback
            import json
            import os
            import warnings
            import contextlib
            import io
            warnings.filterwarnings('ignore')
            os.environ['OPENBLAS_NUM_THREADS'] = '16'

            generated_code = {escaped_code}

            stdout = io.StringIO()
            stderr = io.StringIO()

            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                try:
                    exec(generated_code)
                    status = "completed"
                except Exception:
                    status = "error"
                    stderr.write(traceback.format_exc())
            stdout = stdout.getvalue()
            stderr = stderr.getvalue()
            if len(stdout) > {max_output_characters}:
                stdout = stdout[:{max_output_characters}] + "<output cut>"
            if len(stderr) > {max_output_characters}:
                stderr = stderr[:{max_output_characters}] + "<output cut>"
            if stdout:
                stdout += "\\n"
            if stderr:
                stderr += "\\n"
            output = {{"process_status": status, "stdout": stdout, "stderr": stderr}}
            print(json.dumps(output))
        """).strip()
        request = self._prepare_request(code_to_execute, timeout_seconds)
        try:
            return self._send_request(request, timeout_seconds)
        except requests.exceptions.Timeout:
            return {"process_status": "timeout", "stdout": "", "stderr": "Timed out\n"}


class LocalSandbox(Sandbox):
    """Locally hosted sandbox."""

    def __init__(self, *, uri: HttpUrl):
        super().__init__(uri=uri)

    @override
    def _get_execute_url(self, uri: HttpUrl) -> str:
        return urljoin(str(uri), "execute")

    @override
    def _parse_request_output(self, output: requests.Response) -> dict[str, str]:
        try:
            output_json = output.json()
            assert isinstance(output_json, dict)
            return output_json
        except (requests.exceptions.JSONDecodeError, AssertionError) as e:
            logger.exception("Error parsing output: %s. %s", output.text, e)
            return {'process_status': 'error', 'stdout': '', 'stderr': f'Unknown error: {e} \"{output.text}\"'}

    @override
    def _prepare_request(self,
                         generated_code: str,
                         timeout_seconds: float,
                         language: str = "python",
                         **kwargs) -> dict[str, Any]:
        request = {
            "generated_code": generated_code,
            "timeout": timeout_seconds,
            "language": language,
        }
        return request

    @override
    async def execute_code(
        self,
        generated_code: str,
        timeout_seconds: float = 10.0,
        language: str = "python",
        max_output_characters: int = 1000,
    ) -> dict[str, str]:
        """Override execute_code to bypass the wrapper logic and send user code directly to our server."""

        logger.debug("Raw input generated_code: %s", generated_code)

        # The input appears to be a string representation of a dictionary
        # We need to parse it and extract the actual code
        try:
            # Try to evaluate the string as a Python literal (dictionary)
            import ast
            parsed_dict = ast.literal_eval(generated_code)
            if isinstance(parsed_dict, dict) and 'generated_code' in parsed_dict:
                actual_code = parsed_dict['generated_code']
                assert isinstance(actual_code, str)
                logger.debug("Extracted code from dict: %s...", actual_code[:100])
            else:
                # If it's not a dict or doesn't have the expected key, use as-is
                actual_code = generated_code
                logger.debug("Using code as-is: %s...", actual_code[:100])
        except (ValueError, SyntaxError):
            # If parsing fails, use the input as-is
            actual_code = generated_code
            logger.debug("Failed to parse, using as-is: %s...", actual_code[:100])

        # Clean the actual code more carefully to avoid removing backticks that are part of Python code
        # remove all leading/trailing whitespace -- strip()
        # remove all leading/trailing backticks -- strip("`")
        # may potentially start with python, so just trim from the front.
        POTENTIAL_PREFIXES = ["python"]
        actual_code = actual_code.strip().strip("`")
        for prefix in POTENTIAL_PREFIXES:
            if actual_code.startswith(prefix):
                actual_code = actual_code[len(prefix):]
                break

        # Send the user's code directly to our server without any wrapper logic
        # Our server already handles stdout/stderr capture and error handling
        request = self._prepare_request(actual_code, timeout_seconds, language)
        try:
            return self._send_request(request, timeout_seconds)
        except requests.exceptions.Timeout:
            return {"process_status": "timeout", "stdout": "", "stderr": "Timed out\n"}


class PistonSandbox(Sandbox):
    """Piston sandbox (https://github.com/engineer-man/piston)"""

    @override
    def _get_execute_url(self, uri: HttpUrl) -> str:
        return urljoin(str(uri), "execute")

    @override
    def _parse_request_output(self, output: requests.Response) -> dict[str, str]:
        output_json = output.json()
        assert isinstance(output_json, dict)
        assert 'run' in output_json
        run_json = output_json['run']
        assert isinstance(run_json, dict)
        if run_json["code"] != 0:
            return {'process_status': "error", 'stdout': run_json['stdout'], 'stderr': run_json['stderr']}
        return {'process_status': "completed", 'stdout': run_json['stdout'], 'stderr': run_json['stderr']}

    @override
    def _prepare_request(self, generated_code: str, timeout_seconds: float, **kwargs) -> dict[str, Any]:
        return {
            "language": "py",
            "version": "3.10.0",
            "files": [{
                "content": generated_code,
            }],
            "stdin": "",
            "args": [],
            "run_timeout": timeout_seconds * 1000.0,  # milliseconds
            "compile_memory_limit": -1,
            "run_memory_limit": -1,
        }


def get_sandbox(sandbox_type: str = "local", **kwargs):
    """A helper function to make it easier to set sandbox through cmd."""
    sandboxes = {
        'local': LocalSandbox,
        'piston': PistonSandbox,
    }
    sandbox_class = sandboxes[sandbox_type.lower()]
    return sandbox_class(**kwargs)
