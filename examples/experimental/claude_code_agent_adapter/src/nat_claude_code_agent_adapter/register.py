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

import asyncio
import datetime
import logging
from collections.abc import AsyncGenerator
from pathlib import Path
from typing import Any
from typing import Literal

from pydantic import Field

from nat.builder.builder import Builder
from nat.builder.function_info import FunctionInfo
from nat.cli.register_workflow import register_function
from nat.data_models.agent import AgentBaseConfig
from nat.data_models.api_server import ChatRequest
from nat.data_models.api_server import ChatRequestOrMessage
from nat.data_models.api_server import ChatResponse
from nat.data_models.api_server import ChatResponseChunk
from nat.data_models.api_server import Usage
from nat.data_models.component_ref import LLMRef
from nat.utils.type_converter import GlobalTypeConverter

logger = logging.getLogger(__name__)

PermissionMode = Literal["default", "acceptEdits", "plan", "bypassPermissions", "dontAsk", "auto"]
SettingSource = Literal["user", "project", "local"]


class ClaudeCodeAgentWorkflowConfig(AgentBaseConfig, name="claude_code_agent"):
    """Configuration for the Claude Code Agent SDK workflow."""

    llm_name: LLMRef | None = Field(
        default=None,
        description=("Optional NAT LLM reference. Claude Code Agent SDK manages its own model selection through "
                     "`model`, so this field is accepted for agent config consistency but is not used."))
    description: str = Field(default="Claude Code Agent SDK Workflow",
                             description="The description of this function's use.")

    working_directory: str = Field(default=".", description="Directory passed to the Claude Agent SDK as cwd.")
    permission_mode: PermissionMode | None = Field(default="plan", description="Claude Agent SDK permission mode.")
    model: str | None = Field(default=None, description="Optional Claude model name.")
    append_system_prompt: str | None = Field(default=None,
                                             description="Optional prompt appended to Claude Code's preset.")
    allowed_tools: list[str] = Field(default_factory=list, description="Claude Agent SDK tool allow-list.")
    disallowed_tools: list[str] = Field(default_factory=list, description="Claude Agent SDK tool deny-list.")
    setting_sources: list[SettingSource] | None = Field(
        default_factory=lambda: ["project"],
        description="Filesystem setting sources to load. Use [] to disable user, project, and local settings.")
    additional_directories: list[str] = Field(default_factory=list,
                                              description="Additional directories to allow the SDK agent to access.")

    max_turns: int | None = Field(default=1, ge=1, description="Maximum Claude Agent SDK turns.")
    max_budget_usd: float | None = Field(default=None, ge=0.0, description="Maximum SDK request budget.")
    max_history: int | None = Field(default=15, ge=1, description="Maximum NAT chat messages to include in prompt.")
    timeout_seconds: float = Field(default=120.0, gt=0, description="Overall SDK query timeout.")
    max_output_chars: int = Field(default=12000, gt=0, description="Maximum returned output characters.")


def _clip(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + f"\n\n[truncated to {max_chars} characters]"


def _sdk_message_text(message: Any) -> str:
    content = getattr(message, "content", None)
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts = []
        for block in content:
            text = getattr(block, "text", None)
            if isinstance(text, str):
                parts.append(text)
        return "\n".join(parts)

    return ""


def _nat_message_content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict):
                text = block.get("text")
                if isinstance(text, str):
                    parts.append(text)
            else:
                text = getattr(block, "text", None)
                if isinstance(text, str):
                    parts.append(text)
        return "\n".join(parts)
    return str(content)


def _role_to_text(role: Any) -> str:
    return getattr(role, "value", str(role))


def _build_prompt(message: ChatRequestOrMessage, config: ClaudeCodeAgentWorkflowConfig) -> str:
    if message.is_string:
        return message.input_message or ""

    chat_request = GlobalTypeConverter.get().convert(message, to_type=ChatRequest)
    messages = chat_request.messages[-config.max_history:] if config.max_history else chat_request.messages

    if len(messages) == 1 and _role_to_text(messages[0].role) == "user":
        return _nat_message_content_to_text(messages[0].content)

    prompt_parts = []
    for chat_message in messages:
        role = _role_to_text(chat_message.role)
        content = _nat_message_content_to_text(chat_message.content)
        prompt_parts.append(f"{role}: {content}")
    return "\n\n".join(prompt_parts)


def _usage_for(prompt: str, response: str) -> Usage:
    prompt_tokens = len(prompt.split()) if prompt else 0
    completion_tokens = len(response.split()) if response else 0
    return Usage(prompt_tokens=prompt_tokens,
                 completion_tokens=completion_tokens,
                 total_tokens=prompt_tokens + completion_tokens)


def _as_response(content: str, prompt: str, model: str | None) -> ChatResponse:
    return ChatResponse.from_string(content,
                                    model=model or "claude-code-agent",
                                    created=datetime.datetime.now(datetime.UTC),
                                    usage=_usage_for(prompt, content))


def _build_options_kwargs(config: ClaudeCodeAgentWorkflowConfig) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "cwd": str(Path(config.working_directory).resolve()),
        "allowed_tools": config.allowed_tools,
        "disallowed_tools": config.disallowed_tools,
        "setting_sources": config.setting_sources,
        "add_dirs": config.additional_directories,
        "system_prompt": {
            "type": "preset",
            "preset": "claude_code",
        },
        "tools": {
            "type": "preset",
            "preset": "claude_code",
        },
    }

    if config.permission_mode:
        kwargs["permission_mode"] = config.permission_mode
    if config.model:
        kwargs["model"] = config.model
    if config.max_turns is not None:
        kwargs["max_turns"] = config.max_turns
    if config.max_budget_usd is not None:
        kwargs["max_budget_usd"] = config.max_budget_usd
    if config.append_system_prompt is not None:
        kwargs["system_prompt"]["append"] = config.append_system_prompt

    return {key: value for key, value in kwargs.items() if value is not None}


def _load_sdk_types():
    try:
        from claude_agent_sdk import AssistantMessage
        from claude_agent_sdk import ClaudeAgentOptions
        from claude_agent_sdk import ClaudeSDKError
        from claude_agent_sdk import ResultMessage
        from claude_agent_sdk import query
    except ImportError as import_error:
        raise RuntimeError("The Claude Agent SDK is not installed. Install this example with "
                           "`uv pip install -e examples/experimental/claude_code_agent_adapter`.") from import_error

    return AssistantMessage, ClaudeAgentOptions, ClaudeSDKError, ResultMessage, query


async def _query_claude_code(prompt: str, config: ClaudeCodeAgentWorkflowConfig) -> str:
    AssistantMessage, ClaudeAgentOptions, ClaudeSDKError, ResultMessage, query = _load_sdk_types()

    options = ClaudeAgentOptions(**_build_options_kwargs(config))
    assistant_text: list[str] = []
    final_result: str | None = None

    async def _collect() -> str:
        nonlocal final_result
        try:
            async for sdk_message in query(prompt=prompt, options=options):
                if isinstance(sdk_message, AssistantMessage):
                    text = _sdk_message_text(sdk_message)
                    if text:
                        assistant_text.append(text)
                        if config.verbose:
                            logger.info("Claude Code agent message: %s", _clip(text, config.log_response_max_chars))
                elif isinstance(sdk_message, ResultMessage):
                    final_result = sdk_message.result
        except ClaudeSDKError as sdk_error:
            raise RuntimeError(f"Claude Agent SDK failed: {sdk_error}") from sdk_error

        assistant_output = assistant_text[-1].strip() if assistant_text else ""
        return assistant_output or final_result or ""

    text = await asyncio.wait_for(_collect(), timeout=config.timeout_seconds)
    return _clip(text, config.max_output_chars)


@register_function(config_type=ClaudeCodeAgentWorkflowConfig)
async def claude_code_agent(config: ClaudeCodeAgentWorkflowConfig, _builder: Builder):

    async def _response_fn(chat_request_or_message: ChatRequestOrMessage) -> ChatResponse | str:
        message = GlobalTypeConverter.get().convert(chat_request_or_message, to_type=ChatRequestOrMessage)
        prompt = _build_prompt(message, config)
        content = await _query_claude_code(prompt=prompt, config=config)

        if message.is_string:
            return content
        return _as_response(content, prompt=prompt, model=config.model)

    async def _stream_fn(chat_request_or_message: ChatRequestOrMessage) -> AsyncGenerator[ChatResponseChunk]:
        message = GlobalTypeConverter.get().convert(chat_request_or_message, to_type=ChatRequestOrMessage)
        prompt = _build_prompt(message, config)
        yield ChatResponseChunk.from_string(await _query_claude_code(prompt=prompt, config=config),
                                            model=config.model or "claude-code-agent")

    yield FunctionInfo.create(single_fn=_response_fn, stream_fn=_stream_fn, description=config.description)
