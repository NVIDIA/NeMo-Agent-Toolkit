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
```
"""

import argparse
import asyncio
import json
import sys
import webbrowser

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


def build_http_payload(input_message: str) -> dict:
    """Build an OpenAI-compatible HTTP chat payload."""
    return {
        "messages": [{
            "role": "user",
            "content": input_message,
        }],
        "stream": False,
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
    parser.add_argument("--http-url",
                        default="http://localhost:8000/v1/chat",
                        help="HTTP URL for http mode.")
    return parser.parse_args()


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
    payload = build_http_payload(args.input)
    headers = {"Authorization": f"Bearer {token}"}
    response = httpx.post(args.http_url, json=payload, headers=headers, timeout=120.0)
    response.raise_for_status()
    data = response.json()

    message = (
        data.get("choices", [{}])[0].get("message", {}).get("content")
        if isinstance(data, dict) else None
    )
    if isinstance(message, str) and message.strip():
        print(message)
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
