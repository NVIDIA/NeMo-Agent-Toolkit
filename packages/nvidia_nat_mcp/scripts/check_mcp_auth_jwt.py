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
Test script for JWT-based user identification for MCP authentication flows.

Supports:
- WebSocket: identifies user by `Authorization: Bearer <JWT>` header.
- HTTP: identifies user by `Authorization: Bearer <JWT>` header.

Sample usage:
1. Start the NAT server, for example:
```bash
# Terminal 1
nat serve --config_file examples/MCP/simple_auth_mcp/configs/config-mcp-auth-jira-per-user.yml
```

2. Run WebSocket mode:
```bash
python3 packages/nvidia_nat_mcp/scripts/check_mcp_auth_jwt.py --protocol ws
python3 packages/nvidia_nat_mcp/scripts/check_mcp_auth_jwt.py --protocol ws --user-id Alice \
    --input "What is the status of AIQ-1935?"
```

3. Run HTTP mode:
```bash
python3 packages/nvidia_nat_mcp/scripts/check_mcp_auth_jwt.py --protocol http
python3 packages/nvidia_nat_mcp/scripts/check_mcp_auth_jwt.py --protocol http --user-id Hatter \
    --input "What is the status of AIQ-1935?"
python3 packages/nvidia_nat_mcp/scripts/check_mcp_auth_jwt.py --protocol http --http-endpoint chat-stream
```
"""

import argparse
import asyncio
import json
import sys
import time
import webbrowser
from urllib.parse import urljoin

import httpx
import websockets

try:
    from authlib.jose import jwt
except ImportError as e:
    raise ImportError("authlib is required for check_mcp_auth_jwt. Install with: pip install authlib") from e

USER_ID_1 = "Alice"
USER_ID_2 = "Hatter"
USER_ID_3 = "Rabbit"

INPUT_MESSAGE_1 = "What is the status of AIQ-1935?"
INPUT_MESSAGE_2 = "Summarize AIQ-1935"

# Secret used only to sign a test JWT.
_TEST_JWT_SECRET = b"test-secret-for-mcp-jwt-script"


def make_test_jwt(user_id: str) -> str:
    """Build a JWT with user identity claims for server-side user_id resolution."""
    header = {"alg": "HS256", "typ": "JWT"}
    payload = {"sub": user_id, "name": user_id}
    token = jwt.encode(header, payload, _TEST_JWT_SECRET)
    return token.decode() if isinstance(token, bytes) else token


def build_ws_message(input_message: str) -> dict:
    """Build a WebSocket chat request payload."""
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


def build_http_payload(input_message: str, *, stream: bool = False) -> dict:
    """Build an OpenAI-compatible HTTP chat payload."""
    return {
        "messages": [{
            "role": "user",
            "content": input_message,
        }],
        "stream": stream,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Send JWT-authenticated requests over WebSocket or HTTP.")
    parser.add_argument("--protocol",
                        choices=["ws", "http"],
                        default="ws",
                        help="Transport protocol to use. Defaults to ws.")
    parser.add_argument("--user-id", default=USER_ID_1, help="User ID value for JWT name/sub claims.")
    parser.add_argument("--input", default=INPUT_MESSAGE_1, help="User message to send.")
    parser.add_argument("--ws-url",
                        default="ws://localhost:8000/websocket",
                        help="WebSocket URL for ws mode.")
    parser.add_argument("--http-endpoint",
                        choices=["chat", "chat-stream"],
                        default="chat",
                        help=("Preset HTTP endpoint for http mode. "
                              "'chat' -> /v1/chat, 'chat-stream' -> /v1/chat/stream."))
    parser.add_argument("--http-url",
                        default=None,
                        help="HTTP URL override for http mode. If omitted, uses --http-endpoint preset.")
    parser.add_argument("--open-browser",
                        action=argparse.BooleanOptionalAction,
                        default=True,
                        help="Open OAuth URLs in a browser when interactive auth is required.")
    parser.add_argument("--follow-interactive",
                        action=argparse.BooleanOptionalAction,
                        default=True,
                        help="For /v1/chat, poll execution status until completion.")
    parser.add_argument("--poll-interval-seconds",
                        type=float,
                        default=1.0,
                        help="Polling interval in seconds for interactive HTTP execution status.")
    parser.add_argument("--poll-timeout-seconds",
                        type=float,
                        default=300.0,
                        help="Maximum time to wait for interactive HTTP execution completion.")
    return parser.parse_args()


def _resolve_http_url(args: argparse.Namespace) -> str:
    if args.http_url:
        return args.http_url
    if args.http_endpoint == "chat-stream":
        return "http://localhost:8000/v1/chat/stream"
    return "http://localhost:8000/v1/chat"


def _is_streaming_http_target(args: argparse.Namespace, http_url: str) -> bool:
    return args.http_endpoint == "chat-stream" or http_url.rstrip("/").endswith("/stream")


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
    """Return (done, failed) for an execution status payload."""
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


def _follow_http_interactive(client: httpx.Client, http_url: str, first_payload: dict, args: argparse.Namespace) -> None:
    status_url = _absolute_url(http_url, first_payload.get("status_url"))
    if not status_url:
        print(json.dumps(first_payload, indent=2))
        return

    opened_oauth_states: set[str] = set()
    start = time.monotonic()

    current_payload = first_payload
    while True:
        status = current_payload.get("status")
        if status == "oauth_required":
            auth_url = current_payload.get("auth_url")
            oauth_state = current_payload.get("oauth_state")
            state_key = oauth_state if isinstance(oauth_state, str) else "<none>"
            if args.open_browser and isinstance(auth_url, str) and state_key not in opened_oauth_states:
                webbrowser.open(auth_url)
                opened_oauth_states.add(state_key)
        elif status == "interaction_required":
            print(json.dumps(current_payload, indent=2))
            return

        done, failed = _handle_execution_status_payload(current_payload)
        if done:
            if failed:
                raise RuntimeError("Interactive HTTP execution failed.")
            return

        if time.monotonic() - start > args.poll_timeout_seconds:
            raise TimeoutError(f"Timed out polling execution status after {args.poll_timeout_seconds} seconds.")

        time.sleep(args.poll_interval_seconds)
        status_response = client.get(status_url)
        status_response.raise_for_status()
        current_payload = status_response.json()


async def run_ws(args: argparse.Namespace) -> None:
    """Execute a WebSocket request using a JWT auth header."""
    token = make_test_jwt(args.user_id)
    message = build_ws_message(args.input)
    headers = {"Authorization": f"Bearer {token}"}
    async with websockets.connect(args.ws_url, additional_headers=headers) as ws:
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
    """Execute an HTTP request using a JWT auth header."""
    token = make_test_jwt(args.user_id)
    http_url = _resolve_http_url(args)
    use_streaming = _is_streaming_http_target(args, http_url)
    payload = build_http_payload(args.input, stream=use_streaming)
    headers = {"Authorization": f"Bearer {token}"}

    if use_streaming:
        text_chunks: list[str] = []
        captured_payloads: list[dict] = []
        current_event = "message"
        with httpx.stream("POST", http_url, json=payload, headers=headers, timeout=120.0) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if not line:
                    continue

                if line.startswith("event:"):
                    current_event = line[len("event:"):].strip()
                    continue

                if not line.startswith("data:"):
                    continue

                data_str = line[len("data:"):].strip()
                if data_str == "[DONE]":
                    break

                try:
                    data = json.loads(data_str)
                except json.JSONDecodeError:
                    if data_str:
                        text_chunks.append(data_str)
                    continue

                captured_payloads.append(data)
                if current_event == "error":
                    print(json.dumps(data, indent=2), file=sys.stderr)
                    continue
                if current_event == "oauth_required":
                    auth_url = data.get("auth_url")
                    if args.open_browser and isinstance(auth_url, str):
                        webbrowser.open(auth_url)
                    continue

                chunk = (
                    data.get("choices", [{}])[0].get("delta", {}).get("content")
                    or data.get("choices", [{}])[0].get("message", {}).get("content")
                )
                if isinstance(chunk, str) and chunk:
                    text_chunks.append(chunk)

        final_text = "".join(text_chunks).strip()
        if final_text:
            print(final_text)
        elif captured_payloads:
            print(json.dumps(captured_payloads, indent=2))
        return

    with httpx.Client(headers=headers, timeout=120.0) as client:
        response = client.post(http_url, json=payload)
        response.raise_for_status()
        data = response.json()

        if isinstance(data, dict) and data.get("status") in {"oauth_required", "interaction_required", "running"}:
            if args.follow_interactive:
                _follow_http_interactive(client, http_url, data, args)
            else:
                print(json.dumps(data, indent=2))
            return

        if isinstance(data, dict):
            _print_chat_result(data)
        else:
            print(json.dumps(data, indent=2))


async def main() -> None:
    args = parse_args()
    if args.protocol == "ws":
        await run_ws(args)
    else:
        run_http(args)


if __name__ == "__main__":
    asyncio.run(main())
