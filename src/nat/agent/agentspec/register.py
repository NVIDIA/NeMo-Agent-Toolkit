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

import io
import logging
from typing import Any

from nat.builder.framework_enum import LLMFrameworkEnum
from nat.builder.function_info import FunctionInfo
from nat.cli.register_workflow import register_function
from nat.data_models.api_server import ChatRequest
from nat.data_models.api_server import ChatRequestOrMessage
from nat.data_models.api_server import ChatResponse
from nat.data_models.api_server import Usage
from nat.utils.type_converter import GlobalTypeConverter

from .config import AgentSpecWorkflowConfig
from .config import read_agentspec_payload

logger = logging.getLogger(__name__)


def _to_plain_messages(messages: list[Any]) -> list[dict[str, Any]]:
    plain: list[dict[str, Any]] = []
    for m in messages:
        # Accept either NAT Message models or LangChain BaseMessage dicts
        role = None
        content = None
        if isinstance(m, dict):
            role = m.get("role")
            content = m.get("content")
        else:
            # Try NAT Message model
            if hasattr(m, "role"):
                role = getattr(m.role, "value", None) or str(getattr(m, "role"))
            # Various content shapes
            if hasattr(m, "content"):
                c = getattr(m, "content")
                if isinstance(c, str):
                    content = c
                else:
                    try:
                        buf = io.StringIO()
                        for part in c:
                            if hasattr(part, "text"):
                                buf.write(str(getattr(part, "text")))
                            else:
                                buf.write(str(part))
                        content = buf.getvalue()
                    except Exception:
                        content = str(c)
            # Fallback: LangChain BaseMessage has .type
            if role is None and hasattr(m, "type"):
                role = str(getattr(m, "type"))
            if content is None and hasattr(m, "content"):
                content = str(getattr(m, "content"))
        plain.append({"role": role or "user", "content": content or ""})
    return plain


@register_function(config_type=AgentSpecWorkflowConfig, framework_wrappers=[LLMFrameworkEnum.LANGCHAIN])
async def agent_spec_workflow(config: AgentSpecWorkflowConfig, builder):
    # Lazy import to make the dependency optional unless this workflow is used
    try:
        from langgraph_agentspec_adapter.agentspecloader import AgentSpecLoader  # type: ignore
    except Exception as e:  # pragma: no cover - import error path
        raise ImportError("Agent Spec adapter not installed. Install with: pip install 'nvidia-nat[agentspec]'") from e

    # Build tool registry from NAT tool names if provided
    tools = await builder.get_tools(tool_names=config.tool_names, wrapper_type=LLMFrameworkEnum.LANGCHAIN)
    tool_registry = {getattr(t, "name", f"tool_{i}"): t for i, t in enumerate(tools)} if tools else {}

    fmt, payload = read_agentspec_payload(config)
    loader = AgentSpecLoader(tool_registry=tool_registry, checkpointer=None, config=None)

    # Compile Agent Spec to a LangGraph component
    if fmt == "yaml":
        component = loader.load_yaml(payload)
    else:
        component = loader.load_json(payload)

    async def _response_fn(chat_request_or_message: ChatRequestOrMessage) -> ChatResponse | str:
        from langchain_core.messages import trim_messages  # lazy import with LANGCHAIN wrapper

        from nat.agent.base import AGENT_LOG_PREFIX

        try:
            message = GlobalTypeConverter.get().convert(chat_request_or_message, to_type=ChatRequest)

            # Trim message history
            trimmed = trim_messages(messages=[m.model_dump() for m in message.messages],
                                    max_tokens=config.max_history,
                                    strategy="last",
                                    token_counter=len,
                                    start_on="human",
                                    include_system=True)

            # Best-effort: pass messages in a generic shape expected by adapter graphs
            input_state: dict[str, Any] = {"messages": _to_plain_messages(trimmed)}

            result: Any
            result = await component.ainvoke(input_state)

            # Heuristic extraction of assistant content
            content: str | None = None
            if isinstance(result, dict):
                msgs = result.get("messages")
                if isinstance(msgs, list) and msgs:
                    for entry in reversed(msgs):
                        # LangChain BaseMessage objects have `.type` (e.g., 'ai', 'human') and `.content`
                        if hasattr(entry, "type") and hasattr(entry, "content"):
                            role = getattr(entry, "type", None)
                            if role in ("ai", "assistant", "system"):
                                content = str(getattr(entry, "content", ""))
                                break
                        # Dict-shaped message
                        if isinstance(entry, dict):
                            role = entry.get("role")
                            if role in ("assistant", "system", "ai"):
                                content = str(entry.get("content", ""))
                                break
                if content is None and "output" in result:
                    content = str(result.get("output"))
            if content is None and isinstance(result, str):
                content = result
            if content is None:
                content = str(result)

            prompt_tokens = sum(len(str(msg.content).split()) for msg in message.messages)
            completion_tokens = len(content.split()) if content else 0
            usage = Usage(prompt_tokens=prompt_tokens,
                          completion_tokens=completion_tokens,
                          total_tokens=prompt_tokens + completion_tokens)
            response = ChatResponse.from_string(content, usage=usage)
            if chat_request_or_message.is_string:
                return GlobalTypeConverter.get().convert(response, to_type=str)
            return response
        except Exception as ex:  # pragma: no cover - surface original exception
            logger.error("%s Agent Spec workflow failed: %s", AGENT_LOG_PREFIX, str(ex))
            raise

    yield FunctionInfo.from_fn(_response_fn, description=config.description)
