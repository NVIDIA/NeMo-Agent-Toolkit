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
from collections.abc import AsyncGenerator
from pathlib import Path
from typing import Any

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


class HermesAgentWorkflowConfig(AgentBaseConfig, name="hermes_agent"):
    """Configuration for the Hermes Agent CLI workflow."""

    llm_name: LLMRef | None = Field(
        default=None,
        description=("Optional NAT LLM reference. Hermes manages provider/model selection through local config, "
                     "`provider`, and `model`, so this field is accepted for agent config consistency but is not "
                     "used."))
    description: str = Field(default="Hermes Agent CLI Workflow", description="The description of this function's use.")

    command: str = Field(default="hermes", description="Hermes CLI command or absolute path.")
    command_args: list[str] = Field(default_factory=list,
                                    description=("Additional arguments inserted after `command` and before Hermes "
                                                 "one-shot arguments. Useful for launchers such as `uvx`."))
    working_directory: str = Field(default=".", description="Directory used as the Hermes subprocess cwd.")
    provider: str | None = Field(default=None, description="Optional Hermes provider override.")
    model: str | None = Field(default=None, description="Optional Hermes model override.")
    max_history: int | None = Field(default=15, ge=1, description="Maximum NAT chat messages to include in prompt.")
    timeout_seconds: float = Field(default=300.0, gt=0, description="Overall Hermes CLI timeout.")
    max_output_chars: int = Field(default=12000, gt=0, description="Maximum returned output characters.")
    error_on_empty_output: bool = Field(default=True,
                                        description=("Raise an error when Hermes exits successfully but does not "
                                                     "print a final response."))


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


def _build_prompt(message: ChatRequestOrMessage, config: HermesAgentWorkflowConfig) -> str:
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
                                    model=model or "hermes-agent",
                                    created=datetime.datetime.now(datetime.UTC),
                                    usage=_usage_for(prompt, content))


def _build_hermes_command(config: HermesAgentWorkflowConfig, prompt: str) -> list[str]:
    command = [config.command, *config.command_args, "-z", prompt]
    if config.provider:
        command.extend(["--provider", config.provider])
    if config.model:
        command.extend(["--model", config.model])
    return command


def _diagnose_empty_output(command: list[str]) -> str:
    launcher = " ".join(command[:4])
    return ("Hermes CLI exited successfully but printed no final response text. Hermes one-shot mode suppresses "
            "intermediate logs and only writes the final agent response to stdout, so an empty result usually means "
            "Hermes is not fully configured for a model/provider, provider authentication is missing, or the selected "
            "model returned no final text. Run `uvx --from hermes-agent hermes status`; it should show a concrete "
            "model and an authenticated provider. If it shows `Model: (not set)` or missing credentials, run "
            "`uvx --from hermes-agent hermes model`, `uvx --from hermes-agent hermes auth`, or "
            "`uvx --from hermes-agent hermes setup`, then retry. If you use a specific provider, set both `provider` "
            f"and `model` in this workflow config. Launcher prefix: {launcher}")


async def _run_hermes_agent(prompt: str, config: HermesAgentWorkflowConfig) -> str:
    cwd = Path(config.working_directory).resolve()
    command = _build_hermes_command(config, prompt)
    try:
        process = await asyncio.create_subprocess_exec(*command,
                                                       cwd=str(cwd),
                                                       stdout=asyncio.subprocess.PIPE,
                                                       stderr=asyncio.subprocess.PIPE)
    except FileNotFoundError as error:
        raise RuntimeError(
            f"Could not find Hermes launcher command: {command[0]}. Install uv/uvx for the portable default config, "
            "install Hermes Agent so `hermes` is on PATH for `configs/config-installed.yml`, or set `command` / "
            "`command_args` in the workflow config.") from error

    try:
        stdout_bytes, stderr_bytes = await asyncio.wait_for(process.communicate(), timeout=config.timeout_seconds)
    except TimeoutError as error:
        process.kill()
        await process.wait()
        raise RuntimeError(f"Hermes CLI timed out after {config.timeout_seconds} seconds") from error

    stdout = stdout_bytes.decode(errors="replace").strip()
    stderr = stderr_bytes.decode(errors="replace").strip()
    if process.returncode:
        details = "\n".join(part for part in [stderr, stdout] if part)
        raise RuntimeError(f"Hermes CLI failed with exit code {process.returncode}: {_clip(details, 4000)}")

    if not stdout:
        if stderr:
            raise RuntimeError(f"Hermes CLI produced no stdout. Stderr: {_clip(stderr, 4000)}")
        if config.error_on_empty_output:
            raise RuntimeError(_diagnose_empty_output(command))

    return _clip(stdout, config.max_output_chars)


@register_function(config_type=HermesAgentWorkflowConfig)
async def hermes_agent(config: HermesAgentWorkflowConfig, _builder: Builder):

    async def _response_fn(chat_request_or_message: ChatRequestOrMessage) -> ChatResponse | str:
        message = GlobalTypeConverter.get().convert(chat_request_or_message, to_type=ChatRequestOrMessage)
        prompt = _build_prompt(message, config)
        content = await _run_hermes_agent(prompt=prompt, config=config)

        if message.is_string:
            return content
        return _as_response(content, prompt=prompt, model=config.model)

    async def _stream_fn(chat_request_or_message: ChatRequestOrMessage) -> AsyncGenerator[ChatResponseChunk]:
        message = GlobalTypeConverter.get().convert(chat_request_or_message, to_type=ChatRequestOrMessage)
        prompt = _build_prompt(message, config)
        yield ChatResponseChunk.from_string(await _run_hermes_agent(prompt=prompt, config=config),
                                            model=config.model or "hermes-agent")

    yield FunctionInfo.create(single_fn=_response_fn, stream_fn=_stream_fn, description=config.description)
