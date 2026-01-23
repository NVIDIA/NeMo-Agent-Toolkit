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
Test script to send a websocket message without UI.

This script is used to test the websocket MCP authentication without UI.
- It sends a websocket message to the server and waits for the response.
- It also handles the OAuth consent window if needed.

Sample usage:
1. Start the NAT server, for example:
```bash
# Terminal 1
nat serve --config_file examples/MCP/simple_auth_mcp/configs/config-mcp-auth-jira-per-user.yml
```
2. Run the script to test the websocket MCP authentication without UI:
```bash
# Terminal 2
# Run with default user ID and input message
python3 check_ws_mcp_auth_without_ui.py

# Run with specific user ID and input message
python3 check_ws_mcp_auth_without_ui.py --user-id Alice --input "What is the status of AIQ-1935?"
```
"""

import argparse
import asyncio
import json
import sys
import webbrowser

import websockets

# Sample user IDs
USER_ID_1 = "Alice"
USER_ID_2 = "Hatter"
USER_ID_3 = "Rabbit"

# Sample input messages
INPUT_MESSAGE_1 = "What is the status of AIQ-1935?"
INPUT_MESSAGE_2 = "Summarize AIQ-1935"


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
    parser = argparse.ArgumentParser(description="Send a websocket message without UI.")
    parser.add_argument("--user-id", default=USER_ID_1, help="User ID for the websocket session.")
    parser.add_argument("--input", default=INPUT_MESSAGE_1, help="User message to send.")
    parser.add_argument("--ws-url-template",
                        default="ws://localhost:8000/websocket?session=user-{user_id}",
                        help="Websocket URL template with {user_id} placeholder.")
    return parser.parse_args()


async def main() -> None:
    args = parse_args()
    ws_url = args.ws_url_template.format(user_id=args.user_id)
    message = build_message(args.input)
    async with websockets.connect(ws_url) as ws:
        await ws.send(json.dumps(message))
        response_chunks: list[str] = []
        while True:
            raw = await ws.recv()
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                continue

            if msg.get("type") == "system_interaction_message":
                content = msg.get("content", {})
                if content.get("input_type") == "oauth_consent":
                    url = content.get("text")
                    if url:
                        webbrowser.open(url)
                continue

            if msg.get("type") == "error_message":
                content = msg.get("content", {})
                if isinstance(content, dict):
                    print(f"Error: {content.get('message')}", file=sys.stderr)
                else:
                    print(f"Error: {content}", file=sys.stderr)
                return

            if msg.get("type") == "system_response_message":
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


asyncio.run(main())
