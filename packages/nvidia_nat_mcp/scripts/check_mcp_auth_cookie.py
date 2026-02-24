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
"""
Test script for cookie-based user identification for MCP authentication flows.

Supports:
- WebSocket: identifies user via `?session={user_id}` query parameter.
- HTTP: identifies user via `nat-session` cookie.

Sample usage:
1. Start the NeMo Agent Toolkit server, for example:
```bash
# Terminal 1
nat serve --config_file examples/MCP/simple_auth_mcp/configs/config-mcp-auth-jira-per-user.yml
```

2. Run WebSocket mode:
```bash
python3 packages/nvidia_nat_mcp/scripts/check_mcp_auth_cookie.py --protocol ws
python3 packages/nvidia_nat_mcp/scripts/check_mcp_auth_cookie.py --protocol ws --user-id Alice \
    --input "What is the status of AIQ-1935?"
```

3. Run HTTP mode:
```bash
python3 packages/nvidia_nat_mcp/scripts/check_mcp_auth_cookie.py --protocol http
python3 packages/nvidia_nat_mcp/scripts/check_mcp_auth_cookie.py --protocol http --user-id Hatter \
    --input "What is the status of AIQ-1935?"
```
"""

import argparse
import asyncio
import json
import re
import sys
import time
import webbrowser
from urllib.parse import quote
from urllib.parse import urljoin
from urllib.parse import urlsplit

import httpx
import websockets

USER_ID_1 = "Alice"
USER_ID_2 = "Hatter"
USER_ID_3 = "Rabbit"

INPUT_MESSAGE_1 = "What is the status of AIQ-1935?"
INPUT_MESSAGE_2 = "Summarize AIQ-1935"
_USER_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]+$")


class _InteractiveExecutionError(RuntimeError):
    """Raised when interactive `HTTP` execution fails."""

    def __init__(self) -> None:
        super().__init__("Interactive HTTP execution failed.")


class _ExecutionStatusTimeout(TimeoutError):
    """Raised when execution status polling exceeds timeout."""

    def __init__(self, timeout_seconds: float) -> None:
        super().__init__(f"Timed out polling execution status after {timeout_seconds} seconds.")


def build_ws_message(input_message: str) -> dict:
    """Build a `WebSocket` chat request payload.

    Args:
        input_message: User message to include in the request payload.

    Returns:
        `dict`: A serialized `WebSocket` request payload.
    """
    return {
        "type": "user_message",
        "schema_type": "chat",
        "id": "msg-1",
        "conversation_id": "conv-1",
        "content": {
            "messages": [{
                "role": "user",
                "content": [{
                    "type": "text", "text": input_message
                }],
            }]
        },
    }


def build_http_payload(input_message: str) -> dict:
    """Build an OpenAI-compatible `HTTP` chat payload.

    Args:
        input_message: User message to include in the request payload.

    Returns:
        `dict`: A serialized non-streaming `HTTP` request payload.
    """
    return {
        "messages": [{
            "role": "user",
            "content": input_message,
        }],
        "stream": False,
    }


def parse_args() -> argparse.Namespace:
    """Parse and validate CLI arguments.

    Returns:
        `argparse.Namespace`: Parsed and validated CLI arguments.
    """
    parser = argparse.ArgumentParser(description="Send cookie-authenticated requests over WebSocket or HTTP.")
    parser.add_argument("--protocol",
                        choices=["ws", "http"],
                        default="ws",
                        help="Transport protocol to use. Defaults to ws.")
    parser.add_argument("--user-id", default=USER_ID_1, help="User ID for cookie/session identification.")
    parser.add_argument("--input", default=INPUT_MESSAGE_1, help="User message to send.")
    parser.add_argument("--ws-url-template",
                        default="ws://localhost:8000/websocket?session={user_id}",
                        help="WebSocket URL template with {user_id} placeholder for ws mode.")
    parser.add_argument("--http-endpoint",
                        choices=["chat"],
                        default="chat",
                        help="Preset HTTP endpoint for http mode. Currently supports only 'chat' -> /v1/chat.")
    parser.add_argument("--http-url",
                        default=None,
                        help="HTTP URL override for http mode. If omitted, uses --http-endpoint preset.")
    args = parser.parse_args()

    try:
        args.user_id = _validate_user_id(args.user_id)
        if args.http_url:
            args.http_url = _validate_http_url(args.http_url)
    except argparse.ArgumentTypeError as exc:
        parser.error(str(exc))

    return args


def _validate_user_id(raw_user_id: str) -> str:
    value = raw_user_id.strip()
    if not value:
        raise argparse.ArgumentTypeError("--user-id must not be empty.")
    if not _USER_ID_PATTERN.fullmatch(value):
        raise argparse.ArgumentTypeError("--user-id may contain only letters, numbers, '-' and '_'.")
    return value


def _validate_http_url(raw_url: str) -> str:
    value = raw_url.strip()
    parsed = urlsplit(value)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise argparse.ArgumentTypeError("--http-url must be a valid http/https URL.")
    return value


def _resolve_http_url(args: argparse.Namespace) -> str:
    if args.http_url:
        return args.http_url
    return "http://localhost:8000/v1/chat"


def _absolute_url(http_url: str, maybe_relative: str | None) -> str | None:
    if not maybe_relative:
        return None
    return urljoin(http_url, maybe_relative)


def _print_chat_result(data: dict) -> None:
    message = data.get("choices", [{}])[0].get("message", {}).get("content")
    if isinstance(message, str) and message.strip():
        print(message)
    else:
        print(json.dumps(data, indent=2))


def _handle_execution_status_payload(status_payload: dict) -> tuple[bool, bool]:
    status = status_payload.get("status")
    if status == "completed":
        result = status_payload.get("result")
        if isinstance(result, dict):
            _print_chat_result(result)
        else:
            print(json.dumps(status_payload, indent=2))
        return True, False
    if status == "failed":
        print(json.dumps(status_payload, indent=2), file=sys.stderr)
        return True, True
    return False, False


def _follow_http_interactive(client: httpx.Client, http_url: str, first_payload: dict) -> None:
    status_url = _absolute_url(http_url, first_payload.get("status_url"))
    if not status_url:
        print(json.dumps(first_payload, indent=2))
        return

    opened_oauth_states: set[str] = set()
    start = time.monotonic()
    poll_interval_seconds = 1.0
    poll_timeout_seconds = 300.0

    current_payload = first_payload
    while True:
        status = current_payload.get("status")
        if status == "oauth_required":
            auth_url = current_payload.get("auth_url")
            oauth_state = current_payload.get("oauth_state")
            state_key = oauth_state if isinstance(oauth_state, str) else "<none>"
            if isinstance(auth_url, str) and state_key not in opened_oauth_states:
                webbrowser.open(auth_url)
                opened_oauth_states.add(state_key)
        elif status == "interaction_required":
            print(json.dumps(current_payload, indent=2))
            return

        done, failed = _handle_execution_status_payload(current_payload)
        if done:
            if failed:
                raise _InteractiveExecutionError()
            return

        if time.monotonic() - start > poll_timeout_seconds:
            raise _ExecutionStatusTimeout(poll_timeout_seconds)

        time.sleep(poll_interval_seconds)
        status_response = client.get(status_url)
        status_response.raise_for_status()
        current_payload = status_response.json()


async def run_ws(args: argparse.Namespace) -> None:
    """Execute a `WebSocket` request with a `nat-session` user identifier.

    Args:
        args: Parsed CLI arguments from `parse_args()`.
    """
    safe_user_id = quote(args.user_id, safe="")
    ws_url = args.ws_url_template.format(user_id=safe_user_id)
    message = build_ws_message(args.input)
    async with websockets.connect(ws_url) as ws:
        await ws.send(json.dumps(message))
        response_chunks: list[str] = []
        while True:
            raw = await ws.recv()
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                continue

            match msg.get("type"):
                case "system_interaction_message":
                    content = msg.get("content", {})
                    if content.get("input_type") == "oauth_consent" and (url := content.get("text")):
                        webbrowser.open(url)
                    continue

                case "error_message":
                    content = msg.get("content", {})
                    if isinstance(content, dict):
                        print(f"Error: {content.get('message')}", file=sys.stderr)
                    else:
                        print(f"Error: {content}", file=sys.stderr)
                    return

                case "system_response_message":
                    content = msg.get("content", {})
                    if isinstance(content, dict):
                        chunk = content.get("text") or content.get("output")
                        if isinstance(chunk, str) and msg.get("status") == "in_progress":
                            response_chunks.append(chunk)
                    if msg.get("status") == "complete":
                        final_answer = "".join(response_chunks).strip()
                        if final_answer:
                            print(final_answer)
                        return
                    continue

                case _:
                    continue


def run_http(args: argparse.Namespace) -> None:
    """Execute an `HTTP` request using a `nat-session` cookie.

    Args:
        args: Parsed CLI arguments from `parse_args()`.
    """
    http_url = _resolve_http_url(args)
    payload = build_http_payload(args.input)
    safe_user_id = quote(args.user_id, safe="")
    cookies = {"nat-session": safe_user_id}
    with httpx.Client(cookies=cookies, timeout=120.0) as client:
        response = client.post(http_url, json=payload)
        response.raise_for_status()
        data = response.json()

        if isinstance(data, dict) and data.get("status") in {"oauth_required", "interaction_required", "running"}:
            _follow_http_interactive(client, http_url, data)
            return

        if isinstance(data, dict):
            _print_chat_result(data)
        else:
            print(json.dumps(data, indent=2))


async def main() -> None:
    """Run the selected transport path for cookie-based auth testing."""
    args = parse_args()
    if args.protocol == "ws":
        await run_ws(args)
    else:
        run_http(args)


if __name__ == "__main__":
    asyncio.run(main())
