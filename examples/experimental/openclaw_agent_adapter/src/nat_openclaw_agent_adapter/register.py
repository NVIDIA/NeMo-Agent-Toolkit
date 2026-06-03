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
import json
import os
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

CodexAppServerMode = Literal["guardian", "yolo"]
CodexAppServerApprovalPolicy = Literal["on-request", "on-failure", "untrusted", "never"]
CodexAppServerSandbox = Literal["read-only", "workspace-write", "danger-full-access"]


class OpenClawAgentWorkflowConfig(AgentBaseConfig, name="openclaw_agent"):
    """Configuration for the OpenClaw CLI workflow."""

    llm_name: LLMRef | None = Field(
        default=None,
        description=("Optional NAT LLM reference. OpenClaw manages model selection through local config and `model`, "
                     "so this field is accepted for agent config consistency but is not used."))
    description: str = Field(default="OpenClaw CLI Agent Workflow",
                             description="The description of this function's use.")

    command: str = Field(default="openclaw", description="OpenClaw CLI command or absolute path.")
    working_directory: str = Field(default=".", description="Directory used as the OpenClaw subprocess cwd.")
    agent_id: str = Field(default="main", description="OpenClaw agent id.")
    session_key: str | None = Field(default="nat-openclaw", description="Optional durable OpenClaw session key.")
    model: str | None = Field(default=None, description="Optional OpenClaw model reference.")
    thinking: str | None = Field(default=None, description="Optional OpenClaw thinking level.")
    local: bool = Field(default=False, description="Force embedded local execution instead of Gateway mode.")
    agent_timeout_seconds: int = Field(default=600, gt=0, description="OpenClaw agent run timeout.")
    codex_app_server_mode: CodexAppServerMode | None = Field(
        default="guardian",
        description=("Optional OpenClaw Codex app-server policy mode forwarded through "
                     "OPENCLAW_CODEX_APP_SERVER_MODE."))
    codex_app_server_approval_policy: CodexAppServerApprovalPolicy | None = Field(
        default="on-request",
        description=("Optional OpenClaw Codex app-server approval policy forwarded through "
                     "OPENCLAW_CODEX_APP_SERVER_APPROVAL_POLICY."))
    codex_app_server_sandbox: CodexAppServerSandbox | None = Field(
        default="workspace-write",
        description=("Optional OpenClaw Codex app-server sandbox forwarded through OPENCLAW_CODEX_APP_SERVER_SANDBOX."))

    max_history: int | None = Field(default=15, ge=1, description="Maximum NAT chat messages to include in prompt.")
    timeout_seconds: float = Field(default=660.0, gt=0, description="Overall OpenClaw CLI timeout.")
    max_output_chars: int = Field(default=12000, gt=0, description="Maximum returned output characters.")


def _clip(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + f"\n\n[truncated to {max_chars} characters]"


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


def _build_prompt(message: ChatRequestOrMessage, config: OpenClawAgentWorkflowConfig) -> str:
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
                                    model=model or "openclaw-agent",
                                    created=datetime.datetime.now(datetime.UTC),
                                    usage=_usage_for(prompt, content))


def _build_openclaw_command(config: OpenClawAgentWorkflowConfig, prompt: str) -> list[str]:
    command = [
        config.command,
        "agent",
        "--agent",
        config.agent_id,
        "--message",
        prompt,
        "--timeout",
        str(config.agent_timeout_seconds),
        "--json",
    ]
    if config.session_key:
        command.extend(["--session-key", config.session_key])
    if config.model:
        command.extend(["--model", config.model])
    if config.thinking:
        command.extend(["--thinking", config.thinking])
    if config.local:
        command.append("--local")
    return command


def _add_gateway_path_context(prompt: str, cwd: Path, config: OpenClawAgentWorkflowConfig) -> str:
    if config.local:
        return prompt

    return ("The NeMo Agent Toolkit workflow launched OpenClaw from this directory:\n"
            f"{cwd}\n\n"
            "When the request mentions relative file paths, resolve them against that directory.\n\n"
            f"User request:\n{prompt}")


def _build_openclaw_env(config: OpenClawAgentWorkflowConfig) -> dict[str, str]:
    env = os.environ.copy()
    if config.codex_app_server_mode is not None:
        env["OPENCLAW_CODEX_APP_SERVER_MODE"] = config.codex_app_server_mode
    if config.codex_app_server_approval_policy is not None:
        env["OPENCLAW_CODEX_APP_SERVER_APPROVAL_POLICY"] = config.codex_app_server_approval_policy
    if config.codex_app_server_sandbox is not None:
        env["OPENCLAW_CODEX_APP_SERVER_SANDBOX"] = config.codex_app_server_sandbox
    return env


def _extract_text(value: Any, depth: int = 0) -> str:
    if isinstance(value, str):
        return value
    if value is None or depth > 6:
        return ""
    if isinstance(value, list):
        return "\n".join(part for item in value if (part := _extract_text(item, depth + 1)))
    if not isinstance(value, dict):
        return ""

    for key in ["text", "message", "content", "response", "reply", "finalResponse", "final_response", "output"]:
        text = _extract_text(value.get(key), depth + 1)
        if text:
            return text

    for key in ["payloads", "result", "data"]:
        text = _extract_text(value.get(key), depth + 1)
        if text:
            return text
    return ""


def _extract_openclaw_output(stdout: str, max_chars: int) -> str:
    try:
        result = json.loads(stdout)
    except json.JSONDecodeError:
        return _clip(stdout, max_chars)

    text = _extract_text(result)
    if not text:
        text = json.dumps(result, indent=2, sort_keys=True)
    return _clip(text, max_chars)


def _diagnostic_hint(details: str) -> str:
    if "approval_policy" in details and "Never" in details and "allowed set" in details:
        return ("\n\nHint: OpenClaw is using the Codex app-server harness with a full-access approval policy, but the "
                "Codex cloud requirements for this account reject `Never`. Keep "
                "`codex_app_server_mode: guardian`, `codex_app_server_approval_policy: on-request`, and "
                "`codex_app_server_sandbox: workspace-write`, then use a fresh `session_key` if a previous OpenClaw "
                "session was created with the rejected policy.")
    if "gateway closed" in details or "GatewayTransportError" in details:
        return ("\n\nHint: This Relay example uses OpenClaw Gateway mode so the `nemo-relay` plugin can observe the "
                "run. Restart OpenClaw Gateway, verify `openclaw plugins inspect nemo-relay --runtime --json`, and "
                "check `openclaw gateway call nemoRelay.status --json`. If you only need a direct OpenClaw run "
                "without Relay telemetry, set `local: true`.")
    if "GatewayCredentialsRequiredError" in details or "gateway agent requires credentials" in details:
        return ("\n\nHint: This Relay example uses OpenClaw Gateway mode. Configure local Gateway auth before running "
                "the workflow: set `gateway.mode: local`, `gateway.bind: loopback`, `gateway.auth.mode: token`, and "
                "a `gateway.auth.token`, then restart OpenClaw Gateway.")
    if "timed out waiting for cloud requirements" in details:
        return ("\n\nHint: OpenClaw's Codex harness could not load cloud requirements. Run the workflow from the same "
                "normal shell/user profile that can read and write `~/.openclaw`, verify `openclaw doctor`, and make "
                "sure the selected OpenClaw/Codex model is available for the account.")
    return ""


async def _run_openclaw_agent(prompt: str, config: OpenClawAgentWorkflowConfig) -> str:
    cwd = Path(config.working_directory).resolve()
    openclaw_prompt = _add_gateway_path_context(prompt, cwd, config)
    command = _build_openclaw_command(config, openclaw_prompt)

    try:
        process = await asyncio.create_subprocess_exec(*command,
                                                       cwd=str(cwd),
                                                       env=_build_openclaw_env(config),
                                                       stdout=asyncio.subprocess.PIPE,
                                                       stderr=asyncio.subprocess.PIPE)
    except FileNotFoundError as error:
        raise RuntimeError(f"Could not find OpenClaw CLI command: {command[0]}") from error

    try:
        stdout_bytes, stderr_bytes = await asyncio.wait_for(process.communicate(), timeout=config.timeout_seconds)
    except TimeoutError as error:
        process.kill()
        await process.wait()
        raise RuntimeError(f"OpenClaw CLI timed out after {config.timeout_seconds} seconds") from error

    stdout = stdout_bytes.decode(errors="replace").strip()
    stderr = stderr_bytes.decode(errors="replace").strip()
    if process.returncode:
        details = "\n".join(part for part in [stderr, stdout] if part)
        clipped_details = _clip(details, 4000)
        raise RuntimeError(f"OpenClaw CLI failed with exit code {process.returncode}: {clipped_details}"
                           f"{_diagnostic_hint(details)}")

    return _extract_openclaw_output(stdout, config.max_output_chars)


@register_function(config_type=OpenClawAgentWorkflowConfig)
async def openclaw_agent(config: OpenClawAgentWorkflowConfig, _builder: Builder) -> AsyncGenerator[FunctionInfo, None]:
    """Create the OpenClaw workflow function.

    Args:
        config: OpenClaw workflow configuration used to build subprocess commands and response handling.
        _builder: Workflow builder supplied by NeMo Agent Toolkit. It is accepted for registry compatibility.

    Returns:
        An async generator that yields the FunctionInfo for normal and streaming OpenClaw responses.
    """

    async def _response_fn(chat_request_or_message: ChatRequestOrMessage) -> ChatResponse | str:
        message = GlobalTypeConverter.get().convert(chat_request_or_message, to_type=ChatRequestOrMessage)
        prompt = _build_prompt(message, config)
        content = await _run_openclaw_agent(prompt=prompt, config=config)

        if message.is_string:
            return content
        return _as_response(content, prompt=prompt, model=config.model)

    async def _stream_fn(chat_request_or_message: ChatRequestOrMessage) -> AsyncGenerator[ChatResponseChunk]:
        message = GlobalTypeConverter.get().convert(chat_request_or_message, to_type=ChatRequestOrMessage)
        prompt = _build_prompt(message, config)
        yield ChatResponseChunk.from_string(await _run_openclaw_agent(prompt=prompt, config=config),
                                            model=config.model or "openclaw-agent")

    yield FunctionInfo.create(single_fn=_response_fn, stream_fn=_stream_fn, description=config.description)
