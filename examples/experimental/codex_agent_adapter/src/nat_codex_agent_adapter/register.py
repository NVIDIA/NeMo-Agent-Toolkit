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

"""Register the experimental Codex adapter with NVIDIA NeMo Agent Toolkit."""

import asyncio
import datetime
import json
import os
import shlex
import tempfile
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
from nat.experimental.relay_telemetry_bridge import inject_atof_jsonl
from nat.utils.type_converter import GlobalTypeConverter

ApprovalPolicy = Literal["never", "on-request", "on-failure", "untrusted"]
SandboxMode = Literal["read-only", "workspace-write", "danger-full-access"]
WebSearchMode = Literal["disabled", "cached", "live"]


class CodexAgentWorkflowConfig(AgentBaseConfig, name="codex_agent"):
    """Configuration for the Codex workflow."""

    llm_name: LLMRef | None = Field(
        default=None,
        description=("Optional NAT LLM reference. Codex manages its own model selection through `model`, so this field "
                     "is accepted for agent config consistency but is not used."))
    description: str = Field(default="Codex Agent Workflow", description="The description of this function's use.")

    command: str = Field(default="codex", description="Codex CLI command or absolute path.")
    command_args: list[str] = Field(default_factory=list,
                                    description=("Additional arguments inserted after `command` and before Codex "
                                                 "non-interactive query arguments. Useful for root-level Codex CLI "
                                                 "flags."))
    working_directory: str = Field(default=".", description="Directory used as the Codex subprocess cwd.")

    model: str | None = Field(default=None, description="Optional Codex model name.")
    sandbox_mode: SandboxMode | None = Field(default="read-only", description="Codex sandbox mode.")
    skip_git_repo_check: bool = Field(default=False,
                                      description="Skip the Codex working-directory Git repository check.")
    approval_policy: ApprovalPolicy | None = Field(default="never", description="Codex approval policy.")
    web_search_mode: WebSearchMode | None = Field(default=None, description="Optional web search mode.")
    web_search_enabled: bool | None = Field(default=None, description="Optional legacy web search toggle.")
    additional_directories: list[str] = Field(default_factory=list,
                                              description="Additional directories to expose to Codex.")

    max_history: int | None = Field(default=15, ge=1, description="Maximum NAT chat messages to include in prompt.")
    timeout_seconds: float = Field(default=300.0, gt=0, description="Overall Relay/Codex timeout.")
    max_output_chars: int = Field(default=12000, gt=0, description="Maximum returned output characters.")
    relay_command: str = Field(default="nemo-relay", description="NeMo Relay CLI command or absolute path.")
    relay_atof_output_dir: str | None = Field(
        default=None,
        description=("Optional directory where Relay ATOF JSONL should be persisted. If unset, the adapter uses a "
                     "temporary directory and removes it after importing telemetry."))
    prefer_chatgpt_auth: bool = Field(
        default=True,
        description=("Prefer Codex's stored ChatGPT login over OPENAI_API_KEY for Relay runs. This keeps Codex model "
                     "catalog requests on the ChatGPT Codex backend instead of the public OpenAI models endpoint."))


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


def _build_prompt(message: ChatRequestOrMessage, config: CodexAgentWorkflowConfig) -> str:
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
                                    model=model or "codex-agent",
                                    created=datetime.datetime.now(datetime.UTC),
                                    usage=_usage_for(prompt, content))


def _build_codex_root_args(config: CodexAgentWorkflowConfig) -> list[str]:
    command = []
    if config.approval_policy:
        command.extend(["--ask-for-approval", config.approval_policy])
    return command


def _build_codex_args(config: CodexAgentWorkflowConfig, prompt: str) -> list[str]:
    command = ["exec"]
    if config.model:
        command.extend(["--model", config.model])
    if config.sandbox_mode:
        command.extend(["--sandbox", config.sandbox_mode])
    if config.working_directory:
        command.extend(["--cd", str(Path(config.working_directory).resolve())])
    if config.skip_git_repo_check:
        command.append("--skip-git-repo-check")
    if config.web_search_enabled or config.web_search_mode == "live":
        command.append("--search")
    for directory in config.additional_directories:
        command.extend(["--add-dir", str(Path(directory).resolve())])
    command.extend(["--", prompt])
    return command


def _write_relay_config(config: CodexAgentWorkflowConfig, path: Path) -> None:
    codex_command = shlex.join([config.command, *config.command_args, *_build_codex_root_args(config)])
    path.write_text("[agents.codex]\n"
                    f"command = {json.dumps(codex_command)}\n")


def _relay_plugin_config(atof_dir: Path) -> str:
    return json.dumps({
        "version":
            1,
        "components": [{
            "kind": "observability",
            "enabled": True,
            "config": {
                "atof": {
                    "enabled": True,
                    "output_directory": str(atof_dir),
                    "filename": "events.jsonl",
                    "mode": "overwrite",
                }
            },
        }],
    })


def _build_relay_command(config: CodexAgentWorkflowConfig, prompt: str, relay_config_path: Path,
                         atof_dir: Path) -> list[str]:
    return [
        config.relay_command,
        "run",
        "--agent",
        "codex",
        "--config",
        str(relay_config_path),
        "--plugin-config",
        _relay_plugin_config(atof_dir),
        "--",
        *_build_codex_args(config, prompt),
    ]


def _inject_relay_events(atof_path: Path) -> None:
    if atof_path.exists():
        inject_atof_jsonl(atof_path)


def _build_subprocess_env(config: CodexAgentWorkflowConfig) -> dict[str, str]:
    env = os.environ.copy()
    if config.prefer_chatgpt_auth:
        env.pop("OPENAI_API_KEY", None)
    return env


async def _run_codex_cli(prompt: str, config: CodexAgentWorkflowConfig) -> str:
    cwd = Path(config.working_directory).resolve()
    relay_temp_dir = tempfile.TemporaryDirectory(prefix="nat-codex-relay-")
    relay_root = Path(relay_temp_dir.name)
    relay_config_path = relay_root / "config.toml"
    relay_atof_dir = (Path(config.relay_atof_output_dir).resolve() if config.relay_atof_output_dir else relay_root /
                      "atof")
    relay_atof_dir.mkdir(parents=True, exist_ok=True)
    _write_relay_config(config, relay_config_path)
    relay_atof_path = relay_atof_dir / "events.jsonl"
    command = _build_relay_command(config, prompt, relay_config_path, relay_atof_dir)
    if not cwd.exists() or not cwd.is_dir():
        relay_temp_dir.cleanup()
        raise RuntimeError(f"Invalid working_directory {config.working_directory!r}: {cwd} is not a directory.")

    try:
        process = await asyncio.create_subprocess_exec(*command,
                                                       cwd=str(cwd),
                                                       env=_build_subprocess_env(config),
                                                       stdout=asyncio.subprocess.PIPE,
                                                       stderr=asyncio.subprocess.PIPE)
    except FileNotFoundError as error:
        relay_temp_dir.cleanup()
        raise RuntimeError(
            f"Could not find Codex relay command: {command[0]}. Install NeMo Relay as `nemo-relay` or set "
            "`relay_command` in the workflow config.") from error

    try:
        stdout_bytes, stderr_bytes = await asyncio.wait_for(process.communicate(), timeout=config.timeout_seconds)
    except TimeoutError as error:
        process.kill()
        await process.wait()
        try:
            _inject_relay_events(relay_atof_path)
        finally:
            relay_temp_dir.cleanup()
        raise RuntimeError(f"Codex relay command timed out after {config.timeout_seconds} seconds") from error

    stdout = stdout_bytes.decode(errors="replace").strip()
    stderr = stderr_bytes.decode(errors="replace").strip()
    try:
        _inject_relay_events(relay_atof_path)
    finally:
        relay_temp_dir.cleanup()

    if process.returncode:
        details = "\n".join(part for part in [stderr, stdout] if part)
        raise RuntimeError(f"Codex relay command failed with exit code {process.returncode}: {_clip(details, 4000)}")

    if not stdout and stderr:
        raise RuntimeError(f"Codex relay command produced no stdout. Stderr: {_clip(stderr, 4000)}")

    return _clip(stdout, config.max_output_chars)


async def _run_codex(prompt: str, config: CodexAgentWorkflowConfig) -> str:
    return await _run_codex_cli(prompt, config)


@register_function(config_type=CodexAgentWorkflowConfig)
async def codex_agent(config: CodexAgentWorkflowConfig, _builder: Builder) -> AsyncGenerator[FunctionInfo, None]:
    """Create a Codex workflow function for NVIDIA NeMo Agent Toolkit.

    Args:
        config: Codex workflow configuration from the NeMo Agent Toolkit config system.
        _builder: Toolkit builder supplied during workflow construction.

    Yields:
        FunctionInfo containing single-response and streaming handlers that invoke Codex through NeMo Relay.
    """

    async def _response_fn(chat_request_or_message: ChatRequestOrMessage) -> ChatResponse | str:
        message = GlobalTypeConverter.get().convert(chat_request_or_message, to_type=ChatRequestOrMessage)
        prompt = _build_prompt(message, config)
        content = await _run_codex(prompt=prompt, config=config)

        if message.is_string:
            return content
        return _as_response(content, prompt=prompt, model=config.model)

    async def _stream_fn(chat_request_or_message: ChatRequestOrMessage) -> AsyncGenerator[ChatResponseChunk]:
        message = GlobalTypeConverter.get().convert(chat_request_or_message, to_type=ChatRequestOrMessage)
        prompt = _build_prompt(message, config)
        yield ChatResponseChunk.from_string(await _run_codex(prompt=prompt, config=config),
                                            model=config.model or "codex-agent")

    yield FunctionInfo.create(single_fn=_response_fn, stream_fn=_stream_fn, description=config.description)
