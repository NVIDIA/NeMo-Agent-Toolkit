# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""ToolTalk benchmark workflow.

Replays multi-turn ToolTalk conversations using NAT's LLM with tool calling.
Tool calls are executed against ToolTalk's simulated database backends via ToolExecutor.
"""

import json
import logging

from nat.builder.builder import Builder
from nat.builder.framework_enum import LLMFrameworkEnum
from nat.builder.function_info import FunctionInfo
from nat.cli.register_workflow import register_function

from .config import ToolTalkWorkflowConfig

logger = logging.getLogger(__name__)


def _build_tool_schemas(apis_used, disable_docs: bool = False) -> list[dict]:
    """Convert ToolTalk API classes to OpenAI function-calling tool schemas."""
    tools = []
    for api in apis_used:
        doc = api.to_openai_doc(disable_docs)
        required = doc.pop("required")
        doc["parameters"]["required"] = required
        tools.append({"type": "function", "function": doc})
    return tools


def _build_messages(conversation_history: list[dict], metadata: dict | None = None) -> list[dict]:
    """Convert ToolTalk conversation history to OpenAI chat messages format."""
    system_prompt = ("You are a helpful assistant. Here is some user data:"
                     "\nlocation: {location}"
                     "\ntimestamp: {timestamp}"
                     "\nusername (if logged in): {username}")

    messages = []
    if metadata:
        messages.append({
            "role":
                "system",
            "content":
                system_prompt.format(
                    location=metadata.get("location", "unknown"),
                    timestamp=metadata.get("timestamp", "unknown"),
                    username=metadata.get("username", "unknown"),
                ),
        })

    tool_call_id_counter = 123456789
    for turn in conversation_history:
        if turn["role"] in ("user", "assistant"):
            messages.append({"role": turn["role"], "content": turn["text"]})
        elif turn["role"] == "api":
            tool_call_id = str(tool_call_id_counter)
            messages.append({
                "role":
                    "assistant",
                "content":
                    "",
                "tool_calls": [{
                    "id": tool_call_id,
                    "type": "function",
                    "function": {
                        "name": turn["request"]["api_name"],
                        "arguments": json.dumps(turn["request"]["parameters"]),
                    },
                }],
            })
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call_id,
                "content": json.dumps({
                    "response": turn["response"],
                    "exception": turn["exception"],
                }),
            })
            tool_call_id_counter += 1

    return messages


@register_function(config_type=ToolTalkWorkflowConfig, framework_wrappers=[LLMFrameworkEnum.LANGCHAIN])
async def tooltalk_workflow(config: ToolTalkWorkflowConfig, builder: Builder):
    from langchain_core.messages import AIMessage
    from langchain_core.messages import HumanMessage
    from langchain_core.messages import SystemMessage
    from langchain_core.messages import ToolMessage
    from tooltalk.apis import ALL_APIS
    from tooltalk.apis import APIS_BY_NAME
    from tooltalk.apis import SUITES_BY_NAME
    from tooltalk.evaluation.tool_executor import ToolExecutor

    llm = await builder.get_llm(config.llm_name, wrapper_type=LLMFrameworkEnum.LANGCHAIN)

    async def _run_conversation(input_json: str) -> str:
        """Run a ToolTalk conversation: replay user turns, predict tool calls, execute via ToolExecutor."""
        conversation = json.loads(input_json)
        metadata = conversation["metadata"]
        user_data = conversation.get("user")

        # Select APIs based on api_mode
        if config.api_mode == "exact":
            apis_used = [APIS_BY_NAME[name] for name in conversation["apis_used"]]
        elif config.api_mode == "suite":
            apis_used = [api for suite_name in conversation["suites_used"] for api in SUITES_BY_NAME[suite_name].apis]
        else:
            apis_used = ALL_APIS

        # Build OpenAI tool schemas and bind to LLM
        tool_schemas = _build_tool_schemas(apis_used, config.disable_documentation)
        bound_llm = llm.bind_tools(tool_schemas)

        # Fresh ToolExecutor per conversation
        tool_executor = ToolExecutor(init_database_dir=config.database_dir)

        ground_truth_history = []
        api_history = []

        for turn in conversation["conversation"]:
            if turn["role"] == "user":
                ground_truth_history.append({"role": "user", "text": turn["text"]})
                continue

            if turn["role"] != "assistant":
                raise ValueError(f"Unexpected turn role: {turn['role']}")

            # Reset executor state for this turn's prediction
            tool_executor.init_conversation_state(metadata, api_history, user_data)
            predictions = []
            current_history = ground_truth_history.copy()

            # Prediction loop: call LLM, execute tool calls, repeat until assistant response
            tool_call_count = 0
            while True:
                messages = _build_messages(current_history, metadata)

                # Convert to LangChain message objects
                lc_messages = []
                i = 0
                while i < len(messages):
                    msg = messages[i]
                    if msg["role"] == "system":
                        lc_messages.append(SystemMessage(content=msg["content"]))
                    elif msg["role"] == "user":
                        lc_messages.append(HumanMessage(content=msg["content"]))
                    elif msg["role"] == "assistant":
                        if "tool_calls" in msg and msg["tool_calls"]:
                            lc_messages.append(
                                AIMessage(
                                    content=msg.get("content", ""),
                                    tool_calls=[{
                                        "name": tc["function"]["name"],
                                        "args": json.loads(tc["function"]["arguments"]),
                                        "id": tc["id"],
                                    } for tc in msg["tool_calls"]],
                                ))
                        else:
                            lc_messages.append(AIMessage(content=msg["content"]))
                    elif msg["role"] == "tool":
                        lc_messages.append(ToolMessage(
                            content=msg["content"],
                            tool_call_id=msg["tool_call_id"],
                        ))
                    i += 1

                # Call LLM
                response = await bound_llm.ainvoke(lc_messages)

                if response.tool_calls:
                    # LLM made a tool call — execute it via ToolExecutor
                    tc = response.tool_calls[0]
                    api_name = tc["name"]
                    parameters = tc["args"]

                    if parameters is None:
                        request = {"api_name": api_name, "parameters": None}
                        exec_response = {"response": None, "exception": "Failed to parse API call"}
                    else:
                        request, exec_response = tool_executor.execute_tool(api_name, parameters)

                    prediction = {
                        "role": "api",
                        "request": request,
                        "response": exec_response["response"],
                        "exception": exec_response["exception"],
                    }
                    predictions.append(prediction)
                    tool_call_count += 1

                    # Guard against infinite tool-calling loops
                    if tool_call_count >= config.max_tool_calls_per_turn:
                        logger.warning(
                            "Reached max tool calls (%d) for turn, forcing text response",
                            config.max_tool_calls_per_turn,
                        )
                        predictions.append({
                            "role": "assistant",
                            "text": f"[max tool calls ({config.max_tool_calls_per_turn}) reached]",
                        })
                        break

                    # Add to history for next iteration
                    current_history.append(prediction)
                else:
                    # LLM returned a text response — done with this turn
                    predictions.append({
                        "role": "assistant",
                        "text": str(response.content),
                    })
                    break

            # Store predictions on the turn
            turn["predictions"] = predictions

            # Advance ground truth history
            if "apis" in turn:
                for api in turn["apis"]:
                    api_history.append(api)
                    ground_truth_history.append({
                        "role": "api",
                        "request": api["request"],
                        "response": api["response"],
                        "exception": api["exception"],
                    })
            ground_truth_history.append({"role": "assistant", "text": turn["text"]})

        return json.dumps(conversation)

    yield FunctionInfo.from_fn(_run_conversation, description=config.description)
