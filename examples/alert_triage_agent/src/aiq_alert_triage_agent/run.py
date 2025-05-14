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
"""
Alert Triage HTTP Server

This script implements a Flask-based HTTP server that automates alert triage workflows.
It provides an endpoint that:
1. Accepts POST requests containing monitoring alerts in JSON format
2. Collects alert IDs to track all processed alerts
3. Launches an AIQ Toolkit triage agent for each unique alert
4. The triage agent performs automated investigation using diagnostic tools
   and generates structured reports with root cause analysis

The server acts as the entry point for the alert triage system, enabling automated
handling of monitoring alerts from various sources.

Example Usage:

1. Start the server:
   python run.py --host 0.0.0.0 --port 5000 --env_file /path/to/.env

2. Send a single alert (in a separate terminal):
   curl -X POST http://localhost:5000/alerts \
     -H "Content-Type: application/json" \
     -d '{
        "alert_id": 1,
        "alert_name": "InstanceDown",
        "host_id": "test-instance-1.example.com",
        "severity": "critical",
        "description": "Instance test-instance-1.example.com is not available for scrapping for the last 5m. " \
                      "Please check: - instance is up and running; - monitoring service is in place and running; " \
                      "- network connectivity is ok",
        "summary": "Instance test-instance-1.example.com is down",
        "timestamp": "2025-04-28T05:00:00.000000"
     }'

3. Send multiple alerts (in a separate terminal):
   curl -X POST http://localhost:5000/alerts \
     -H "Content-Type: application/json" \
     -d '[{
        "alert_id": 1,
        "alert_name": "InstanceDown",
        "host_id": "test-instance-1.example.com",
        "severity": "critical",
        "description": "Instance test-instance-1.example.com is not available for scrapping for the last 5m. " \
                      "Please check: - instance is up and running; - monitoring service is in place and running; " \
                      "- network connectivity is ok",
        "summary": "Instance test-instance-1.example.com is down",
        "timestamp": "2025-04-28T05:00:00.000000"
     }, {
        "alert_id": 2,
        "alert_name": "CPUUsageHighError",
        "host_id": "test-instance-2.example.com",
        "severity": "critical",
        "description": "CPU Overall usage on test-instance-2.example.com is high ( current value 100% ). " \
                      "Please check: - trend of cpu usage for all cpus; - running processes for investigate issue; " \
                      "- is there any hardware related issues (e.g. IO bottleneck)",
        "summary": "CPU Usage on test-instance-2.example.com is high (error state)",
        "timestamp": "2025-04-28T06:00:00.000000"
     }]'

Response format:
{
    "received_alert_count": 2,  // number of alerts received in the latest request
    "total_launched": 5  // cumulative count of all alerts processed
}
"""

import argparse
import json
import subprocess

from flask import Flask
from flask import jsonify
from flask import request

app = Flask(__name__)


@app.after_request
def apply_hsts(response):
    # Tell browsers to only use HTTPS for the next year, on all sub‑domains, and enable preload
    response.headers['Strict-Transport-Security'] = 'max-age=31536000; includeSubDomains; preload'
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'SAMEORIGIN'
    response.headers['X-XSS-Protection'] = '1; mode=block'

    return response


processed_alerts = []
# will be set in __main__
ENV_FILE = None


def start_process(alert: dict, env_file: str) -> None:
    """
    Launch the external triage agent process with the alert payload.

    :param alert: Dictionary of alert metric labels
    :param env_file: Path to the .env file
    """
    payload = "Here is the alert in JSON format to investigate:\n" + json.dumps(alert)
    cmd = [
        "dotenv",
        "-f",
        env_file,
        "run",
        "aiq",
        "run",
        "--config_file=examples/alert_triage_agent/src/aiq_alert_triage_agent/configs/config.yml",
        "--input",
        payload,
    ]
    try:
        print(
            f"[start_process] Launching triage for alert '{alert.get('alert_name')}' on host '{alert.get('host_id')}'")
        with subprocess.Popen(cmd) as process:
            process.wait()
    except Exception as e:
        print(f"[start_process] Failed to start process: {e}")


@app.route("/alerts", methods=["POST"])
def receive_alert():
    """
    HTTP endpoint to receive a JSON alert via POST.
    Expects application/json with a single alert dict or a list of alerts.
    """
    try:
        data = request.get_json(force=True)
    except Exception:
        return jsonify({"error": "Invalid JSON"}), 400

    alerts = data if isinstance(data, list) else [data]

    for alert in alerts:
        alert_id = alert.get('alert_id')
        processed_alerts.append(alert_id)
        # use the globals-set ENV_FILE
        if ENV_FILE is None:
            raise ValueError("ENV_FILE must be set before processing alerts")
        start_process(alert, ENV_FILE)

    return jsonify({"received_alert_count": len(alerts), "total_launched": len(processed_alerts)}), 200


def parse_args():
    """
    Parse command-line arguments for server configuration.
    """
    parser = argparse.ArgumentParser(description="Run an HTTP server to accept alert POSTs and trigger triage.")
    parser.add_argument("--host", default="0.0.0.0", help="Host/IP to bind to (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=5000, help="Port to listen on (default: 5000)")
    parser.add_argument("--env_file", default=".env", help="Path to the .env file (default: .env)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    # set the global ENV_FILE for use in the Flask handler
    ENV_FILE = args.env_file

    print("---------------[ Alert Triage HTTP Server ]-----------------")
    print("Protocol   : HTTP")
    print(f"Listening  : {args.host}:{args.port}")
    print(f"Env File   : {args.env_file}")
    print("Endpoint   : POST /alerts with JSON payload\n")

    # Start the Flask development server
    app.run(host=args.host, port=args.port)
