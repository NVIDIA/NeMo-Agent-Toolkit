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
"""Register the experimental Cursor Agent adapter with NVIDIA NeMo Agent Toolkit."""

import asyncio
import datetime
import json
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

CursorMode = Literal["plan", "ask"]
SandboxMode = Literal["enabled", "disabled"]


class CursorAgentWorkflowConfig(AgentBaseConfig, name="cursor_agent"):
    """Configuration for the Cursor Agent CLI workflow."""

    llm_name: LLMRef | None = Field(
        default=None,
        description=("Optional NAT LLM reference. Cursor Agent manages model selection through `model`, so this "
                     "field is accepted for agent config consistency but is not used."))
    description: str = Field(default="Cursor Agent CLI Workflow", description="The description of this function's use.")

    command: str = Field(default="cursor-agent", description="Cursor Agent CLI command or absolute path.")
    command_args: list[str] = Field(default_factory=list,
                                    description=("Additional arguments inserted after `command` and before Cursor "
                                                 "Agent non-interactive query arguments."))
    working_directory: str = Field(default=".", description="Directory used as the Cursor Agent workspace.")
    mode: CursorMode | None = Field(default="plan", description="Cursor Agent execution mode.")
    model: str | None = Field(default=None, description="Optional Cursor Agent model name.")
    sandbox: SandboxMode | None = Field(default=None, description="Optional Cursor Agent sandbox setting.")
    trust_workspace: bool = Field(
        default=False,
        description=("Pass `--trust` for non-interactive runs after the workspace has been reviewed and trusted."))
    max_history: int | None = Field(default=15, ge=1, description="Maximum NAT chat messages to include in prompt.")
    timeout_seconds: float = Field(default=120.0, gt=0, description="Overall Cursor Agent CLI timeout.")
    max_output_chars: int = Field(default=12000, gt=0, description="Maximum returned output characters.")
    relay_command: str = Field(default="nemo-relay", description="NeMo Relay CLI command or absolute path.")
    relay_atof_output_dir: str | None = Field(
        default=None,
        description=("Optional directory where Relay ATOF JSONL should be persisted. If unset, the adapter uses a "
                     "temporary directory and removes it after importing telemetry."))
    relay_patch_restore_hooks: bool = Field(
        default=True,
        description=("Allow NeMo Relay to temporarily merge Cursor hook entries into project `.cursor/hooks.json` "
                     "and restore the original file after the run."))


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


def _build_prompt(message: ChatRequestOrMessage, config: CursorAgentWorkflowConfig) -> str:
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
                                    model=model or "cursor-agent",
                                    created=datetime.datetime.now(datetime.UTC),
                                    usage=_usage_for(prompt, content))


def _build_cursor_args(config: CursorAgentWorkflowConfig, prompt: str) -> list[str]:
    command = [
        "--print",
        "--output-format",
        "text",
        "--workspace",
        str(Path(config.working_directory).resolve()),
    ]
    if config.mode:
        command.extend(["--mode", config.mode])
    if config.sandbox:
        command.extend(["--sandbox", config.sandbox])
    if config.trust_workspace:
        command.append("--trust")
    if config.model:
        command.extend(["--model", config.model])
    command.append(prompt)
    return command


def _write_relay_config(config: CursorAgentWorkflowConfig, path: Path) -> None:
    cursor_command = shlex.join([config.command, *config.command_args])
    path.write_text("[agents.cursor]\n"
                    f"command = {json.dumps(cursor_command)}\n"
                    f"patch_restore_hooks = {str(config.relay_patch_restore_hooks).lower()}\n")


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


def _build_relay_command(config: CursorAgentWorkflowConfig, prompt: str, relay_config_path: Path,
                         atof_dir: Path) -> list[str]:
    return [
        config.relay_command,
        "run",
        "--agent",
        "cursor",
        "--config",
        str(relay_config_path),
        "--plugin-config",
        _relay_plugin_config(atof_dir),
        "--",
        *_build_cursor_args(config, prompt),
    ]


def _inject_relay_events(atof_path: Path) -> None:
    if atof_path.exists():
        inject_atof_jsonl(atof_path)


def _cursor_auth_hint(command_name: str) -> str:
    return (f"Cursor Agent CLI is not authenticated. Run `{command_name} login` and "
            f"`{command_name} status` from the same shell that runs `nat`, or export `CURSOR_API_KEY` before "
            "starting the workflow. Cursor app login may not be visible to the standalone agent CLI.")


def _cursor_trust_hint() -> str:
    return ("Cursor Agent CLI needs workspace trust for headless `--print` runs. Review the workspace, then set "
            "`trust_workspace: true` in the workflow config or run Cursor Agent interactively once to trust it.")


def _cursor_sandbox_hint() -> str:
    return ("Cursor Agent sandboxing is not available on this system. Keep `mode: plan` for read-only planning and "
            "set `sandbox: disabled`, or omit `sandbox` to use the local Cursor Agent default.")


async def _run_cursor_agent(prompt: str, config: CursorAgentWorkflowConfig) -> str:
    cwd = Path(config.working_directory).resolve()
    relay_temp_dir = tempfile.TemporaryDirectory(prefix="nat-cursor-relay-")
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
                                                       stdout=asyncio.subprocess.PIPE,
                                                       stderr=asyncio.subprocess.PIPE)
    except FileNotFoundError as error:
        relay_temp_dir.cleanup()
        raise RuntimeError(
            f"Could not find NeMo Relay CLI command: {command[0]}. Install NeMo Relay as `nemo-relay` or set "
            "`relay_command` in the workflow config. Also install Cursor Agent as `cursor-agent` or set `command` / "
            "`command_args` in the workflow config.") from error

    try:
        stdout_bytes, stderr_bytes = await asyncio.wait_for(process.communicate(), timeout=config.timeout_seconds)
    except TimeoutError as error:
        process.kill()
        await process.wait()
        try:
            _inject_relay_events(relay_atof_path)
        finally:
            relay_temp_dir.cleanup()
        raise RuntimeError(f"Cursor Agent CLI timed out after {config.timeout_seconds} seconds") from error

    stdout = stdout_bytes.decode(errors="replace").strip()
    stderr = stderr_bytes.decode(errors="replace").strip()
    try:
        _inject_relay_events(relay_atof_path)
    finally:
        relay_temp_dir.cleanup()

    if process.returncode:
        details = "\n".join(part for part in [stderr, stdout] if part)
        if "Authentication required" in details or "CURSOR_API_KEY" in details:
            details = f"{details}\n\n{_cursor_auth_hint(config.command)}"
        if "Workspace Trust Required" in details:
            details = f"{details}\n\n{_cursor_trust_hint()}"
        if "Sandbox mode is enabled but not available" in details:
            details = f"{details}\n\n{_cursor_sandbox_hint()}"
        raise RuntimeError(f"Cursor Agent CLI failed with exit code {process.returncode}: {_clip(details, 4000)}")

    return _clip(stdout, config.max_output_chars)


@register_function(config_type=CursorAgentWorkflowConfig)
async def cursor_agent(config: CursorAgentWorkflowConfig, _builder: Builder) -> AsyncGenerator[FunctionInfo, None]:
    """Create a Cursor Agent workflow function for NVIDIA NeMo Agent Toolkit.

    Args:
        config: Cursor Agent workflow configuration from the NeMo Agent Toolkit config system.
        _builder: Toolkit builder supplied during workflow construction.

    Yields:
        FunctionInfo containing single-response and streaming handlers that invoke Cursor Agent through NeMo Relay.
    """

    async def _response_fn(chat_request_or_message: ChatRequestOrMessage) -> ChatResponse | str:
        message = GlobalTypeConverter.get().convert(chat_request_or_message, to_type=ChatRequestOrMessage)
        prompt = _build_prompt(message, config)
        content = await _run_cursor_agent(prompt=prompt, config=config)

        if message.is_string:
            return content
        return _as_response(content, prompt=prompt, model=config.model)

    async def _stream_fn(chat_request_or_message: ChatRequestOrMessage) -> AsyncGenerator[ChatResponseChunk]:
        message = GlobalTypeConverter.get().convert(chat_request_or_message, to_type=ChatRequestOrMessage)
        prompt = _build_prompt(message, config)
        yield ChatResponseChunk.from_string(await _run_cursor_agent(prompt=prompt, config=config),
                                            model=config.model or "cursor-agent")

    yield FunctionInfo.create(single_fn=_response_fn, stream_fn=_stream_fn, description=config.description)
