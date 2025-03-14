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
from urllib.parse import urljoin

import requests
from pydantic import HttpUrl

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
        self.url = self._get_execute_url(uri)
        session = requests.Session()
        adapter = requests.adapters.HTTPAdapter(pool_maxsize=1500, pool_connections=1500, max_retries=3)
        session.mount('http://', adapter)
        session.mount('https://', adapter)
        self.http_session = session

    def _send_request(self, request, timeout):
        output = self.http_session.post(
            url=self.url,
            data=json.dumps(request),
            timeout=timeout,
            headers={"Content-Type": "application/json"},
        )
        # retrying 502 errors
        if output.status_code == 502:
            raise requests.exceptions.Timeout

        return self._parse_request_output(output)

    @abc.abstractmethod
    def _parse_request_output(self, output):
        pass

    @abc.abstractmethod
    def _get_execute_url(self, uri):
        pass

    @abc.abstractmethod
    def _prepare_request(self, generated_code, timeout):
        pass

    async def execute_code(
        self,
        generated_code: str,
        timeout: float = 10.0,
        language: str = "python",
        max_output_characters: int = 1000,
    ) -> tuple[dict, str]:

        generated_code = generated_code.lstrip().rstrip().lstrip("`").rstrip("`")
        code_to_execute = """
import traceback
import json
import os
import warnings
import contextlib
import io
warnings.filterwarnings('ignore')
os.environ['OPENBLAS_NUM_THREADS'] = '16'
"""

        code_to_execute += f"""
\ngenerated_code = {repr(generated_code)}\n
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
"""
        request = self._prepare_request(code_to_execute, timeout)
        try:
            output = self._send_request(request, timeout)
        except requests.exceptions.Timeout:
            output = {"process_status": "timeout", "stdout": "", "stderr": "Timed out\n"}
        return output


class LocalSandbox(Sandbox):
    """Locally hosted sandbox."""

    def _get_execute_url(self, uri):
        return urljoin(str(uri), "execute")

    def _parse_request_output(self, output):
        try:
            return output.json()
        except json.JSONDecodeError as e:
            logger.exception("Error  parsing output: %s. %s", output.text, e)
            return {'process_status': 'error', 'stdout': '', 'stderr': 'Unknown error'}

    def _prepare_request(self, generated_code, timeout, language='python', **kwargs):
        return {
            "generated_code": generated_code,
            "timeout": timeout,
            "language": language,
        }


class PistonSandbox(Sandbox):
    """Piston sandbox (https://github.com/engineer-man/piston)"""

    def _get_execute_url(self, uri):
        return urljoin(str(uri), "execute")

    def _parse_request_output(self, output):
        output = output.json()
        if output['run']['signal'] == "SIGKILL":
            return {'result': None, 'error_message': 'Unknown error: SIGKILL'}
        return json.loads(output['run']['output'])

    def _prepare_request(self, generated_code: str, timeout, **kwargs):
        return {
            "language": "py",
            "version": "3.10.0",
            "files": [{
                "content": generated_code,
            }],
            "stdin": "",
            "args": [],
            "run_timeout": timeout * 1000.0,  # milliseconds
            "compile_memory_limit": -1,
            "run_memory_limit": -1,
        }


sandboxes = {
    'local': LocalSandbox,
    'piston': PistonSandbox,
}


def get_sandbox(sandbox_type: str = "local", **kwargs):
    """A helper function to make it easier to set sandbox through cmd."""
    sandbox_class = sandboxes[sandbox_type.lower()]
    return sandbox_class(**kwargs)
