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

import logging
import multiprocessing
import resource
import sys
import os
import pathlib
from io import StringIO

from flask import Flask
from flask import request

app = Flask(__name__)


@app.after_request
def add_hsts_header(response):
    response.headers['Strict-Transport-Security'] = 'max-age=31536000; includeSubDomains'
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'SAMEORIGIN'
    response.headers['X-XSS-Protection'] = '1; mode=block'

    return response


def execute_python(generated_code, timeout):
    # running in a separate process to ensure any kind of crashes are properly handled
    queue = multiprocessing.Queue()
    process = multiprocessing.Process(target=execute_code_subprocess, args=(generated_code, queue))
    
    process.start()
    process.join(timeout=timeout)

    if process.is_alive():  # didn't finish successfully
        process.kill()
        return {"process_status": "timeout", "stdout": "", "stderr": "Timed out\n"}

    result = queue.get()
    return result


# need to memory-limit to avoid common errors of allocating too much
# but this has to be done in a subprocess to not crush server itself
def execute_code_subprocess(generated_code, queue):
    print(f"[DEBUG] execute_code_subprocess started, PID: {os.getpid()}")
    
    limit = 1024 * 1024 * 1024 * 10  # 10gb - somehow with a smaller limit the server dies when numpy is used
    resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
    resource.setrlimit(resource.RLIMIT_DATA, (limit, limit))

    # Capture both stdout and stderr
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    stdout_capture = StringIO()
    stderr_capture = StringIO()
    
    try:
        sys.stdout = stdout_capture
        sys.stderr = stderr_capture
        exec(generated_code, {})  # pylint: disable=W0122
        
        stdout_content = stdout_capture.getvalue()
        stderr_content = stderr_capture.getvalue()
        
        print(f"[DEBUG] Code executed successfully")
        print(f"[DEBUG] stdout length: {len(stdout_content)}")
        print(f"[DEBUG] stderr length: {len(stderr_content)}")
        
        queue.put({
            "process_status": "completed",
            "stdout": stdout_content,
            "stderr": stderr_content
        })
    except Exception as e:
        print(f"[DEBUG] Exception during code execution: {str(e)}")
        stderr_capture.write(f"Error: {str(e)}")
        queue.put({
            "process_status": "error", 
            "stdout": stdout_capture.getvalue(), 
            "stderr": stderr_capture.getvalue()
        })
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr
        print("[DEBUG] Restored stdout/stderr")


# Main Flask endpoint to handle execution requests
@app.route("/execute", methods=["POST"])
def execute():
    try:
        # Check if request has JSON data
        if not request.is_json:
            return {"process_status": "error", "stdout": "", "stderr": "Request must be JSON"}, 400
        
        # Get JSON data safely
        json_data = request.get_json()
        
        if json_data is None:
            return {"process_status": "error", "stdout": "", "stderr": "Invalid JSON data"}, 400
        
        # Check for required fields
        if 'generated_code' not in json_data:
            return {"process_status": "error", "stdout": "", "stderr": "Missing required field: generated_code"}, 400
        
        if 'timeout' not in json_data:
            return {"process_status": "error", "stdout": "", "stderr": "Missing required field: timeout"}, 400
        
        generated_code = json_data['generated_code']
        timeout = json_data['timeout']
        language = json_data.get('language', 'python')

        if language == 'python':
            result = execute_python(generated_code, timeout)
            return result
        
        return {"process_status": "error", "stdout": "", "stderr": "Only python execution is supported"}
    
    except Exception as e:
        import traceback
        return {"process_status": "error", "stdout": "", "stderr": f"Server error: {str(e)}"}, 500


if __name__ == '__main__':
    log = logging.getLogger('werkzeug')
    log.setLevel(logging.WARNING)
    app.run(port=6000)
