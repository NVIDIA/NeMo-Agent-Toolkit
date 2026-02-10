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
Test script to send a websocket message using JWT for user identification (no nat-session cookie).

This script identifies the user via Authorization: Bearer <JWT> and does not send the session
query param, so the server resolves user_id from the JWT payload (name/email/sub claims).
Use this script to test JWT-based user identification.
Complements check_ws_mcp_auth_cookie.py (cookie-based identification via ?session={user_id}).

Sample usage:
1. Start the NAT server, for example:
```bash
# Terminal 1
nat serve --config_file examples/MCP/simple_auth_mcp/configs/config-mcp-auth-jira-per-user.yml
```
2. Run the script to test JWT-based user identification:
```bash
# Terminal 2
# Run with default user ID (Alice) and input message
python3 packages/nvidia_nat_mcp/scripts/check_ws_mcp_auth_jwt.py

# Run with specific user ID and input message
python3 packages/nvidia_nat_mcp/scripts/check_ws_mcp_auth_jwt.py --user-id Alice \
    --input "What is the status of AIQ-1935?"
python3 packages/nvidia_nat_mcp/scripts/check_ws_mcp_auth_jwt.py --user-id Hatter \
    --input "What is the status of AIQ-1935?"
```
"""

import argparse
import asyncio
import json
import sys
import webbrowser

import websockets

try:
    from authlib.jose import jwt
except ImportError as e:
    raise ImportError("authlib is required for check_ws_mcp_auth_jwt. Install with: pip install authlib") from e

# Sample user IDs (same as check_ws_mcp_auth_cookie.py)
USER_ID_1 = "Alice"
USER_ID_2 = "Hatter"
USER_ID_3 = "Rabbit"

# Sample input messages
INPUT_MESSAGE_1 = "What is the status of AIQ-1935?"
INPUT_MESSAGE_2 = "Summarize AIQ-1935"

# Secret used only to sign the test JWT (server does not verify; it only decodes the payload)
_TEST_JWT_SECRET = b"test-secret-for-ws-mcp-jwt-script"


def make_test_jwt(user_id: str) -> str:
    """Build a JWT whose payload includes user identity claims (name, sub) for server-side user_id resolution."""
    header = {"alg": "HS256", "typ": "JWT"}
    payload = {"sub": user_id, "name": user_id}
    token = jwt.encode(header, payload, _TEST_JWT_SECRET)
    return token.decode() if isinstance(token, bytes) else token


def build_message(input_message: str) -> dict:
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Send a websocket message using JWT for user identification (no nat-session cookie).")
    parser.add_argument("--user-id", default=USER_ID_1, help="User ID (put in JWT name/sub claims).")
    parser.add_argument("--input", default=INPUT_MESSAGE_1, help="User message to send.")
    parser.add_argument("--ws-url",
                        default="ws://localhost:8000/websocket",
                        help="Websocket URL (no session query param; user is identified by JWT).")
    return parser.parse_args()


async def main() -> None:
    args = parse_args()
    token = make_test_jwt(args.user_id)
    message = build_message(args.input)
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


if __name__ == "__main__":
    asyncio.run(main())
