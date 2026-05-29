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
from collections.abc import AsyncGenerator
from importlib.resources import files
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

ApprovalPolicy = Literal["never", "on-request", "on-failure", "untrusted"]
ModelReasoningEffort = Literal["minimal", "low", "medium", "high", "xhigh"]
SandboxMode = Literal["read-only", "workspace-write", "danger-full-access"]
WebSearchMode = Literal["disabled", "cached", "live"]


class CodexAgentWorkflowConfig(AgentBaseConfig, name="codex_agent"):
    """Configuration for the Codex SDK workflow."""

    llm_name: LLMRef | None = Field(
        default=None,
        description=("Optional NAT LLM reference. Codex SDK manages its own model selection through `model`, so "
                     "this field is accepted for agent config consistency but is not used."))
    description: str = Field(default="Codex SDK Agent Workflow", description="The description of this function's use.")

    node_command: str = Field(default="node", description="Node.js command or absolute path.")
    node_package_directory: str | None = Field(
        default=None,
        description=("Optional directory containing the adapter package.json/node_modules. Use this when the Node "
                     "SDK dependencies are not installed beside the example source tree."))
    working_directory: str = Field(default=".", description="Directory passed to Codex SDK as workingDirectory.")
    codex_path_override: str | None = Field(default=None, description="Optional local Codex binary override.")
    base_url: str | None = Field(default=None, description="Optional OpenAI-compatible base URL override.")
    codex_config: dict[str, Any] = Field(default_factory=dict, description="Additional Codex SDK config overrides.")

    thread_id: str | None = Field(default=None, description="Optional existing Codex thread id to resume.")
    model: str | None = Field(default=None, description="Optional Codex model name.")
    sandbox_mode: SandboxMode | None = Field(default="read-only", description="Codex SDK sandbox mode.")
    skip_git_repo_check: bool = Field(default=False, description="Skip the SDK working-directory Git repository check.")
    approval_policy: ApprovalPolicy | None = Field(default="never", description="Codex SDK approval policy.")
    model_reasoning_effort: ModelReasoningEffort | None = Field(default=None,
                                                                description="Optional model reasoning effort.")
    network_access_enabled: bool | None = Field(default=None,
                                                description="Optional network access override for the Codex thread.")
    web_search_mode: WebSearchMode | None = Field(default=None, description="Optional web search mode.")
    web_search_enabled: bool | None = Field(default=None, description="Optional legacy web search toggle.")
    additional_directories: list[str] = Field(default_factory=list,
                                              description="Additional directories to expose to Codex.")

    max_history: int | None = Field(default=15, ge=1, description="Maximum NAT chat messages to include in prompt.")
    timeout_seconds: float = Field(default=300.0, gt=0, description="Overall Codex SDK timeout.")
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


def _runner_path() -> Path:
    return Path(str(files(__package__).joinpath("codex_sdk_runner.mjs")))


def _module_search_paths(config: CodexAgentWorkflowConfig, runner: Path) -> list[str]:
    paths: list[Path] = [Path(config.working_directory).resolve()]
    if config.node_package_directory:
        paths.append(Path(config.node_package_directory).resolve())
    for parent in runner.parents:
        if (parent / "package.json").exists():
            paths.append(parent)
            break
    return [str(path) for path in dict.fromkeys(paths)]


def _build_payload(prompt: str, config: CodexAgentWorkflowConfig, runner: Path) -> dict[str, Any]:
    return {
        "prompt": prompt,
        "timeoutMs": max(1000, int(config.timeout_seconds * 1000) - 1000),
        "moduleSearchPaths": _module_search_paths(config, runner),
        "codexOptions": {
            "codexPathOverride": config.codex_path_override,
            "baseUrl": config.base_url,
            "config": config.codex_config or None,
        },
        "threadId": config.thread_id,
        "threadOptions": {
            "model": config.model,
            "sandboxMode": config.sandbox_mode,
            "workingDirectory": str(Path(config.working_directory).resolve()),
            "skipGitRepoCheck": config.skip_git_repo_check,
            "approvalPolicy": config.approval_policy,
            "modelReasoningEffort": config.model_reasoning_effort,
            "networkAccessEnabled": config.network_access_enabled,
            "webSearchMode": config.web_search_mode,
            "webSearchEnabled": config.web_search_enabled,
            "additionalDirectories": config.additional_directories or None,
        },
    }


async def _run_node_sdk(prompt: str, config: CodexAgentWorkflowConfig) -> str:
    runner = _runner_path()
    payload = json.dumps(_build_payload(prompt, config, runner)).encode()
    cwd = Path(config.working_directory).resolve()

    try:
        process = await asyncio.create_subprocess_exec(config.node_command,
                                                       str(runner),
                                                       cwd=str(cwd),
                                                       stdin=asyncio.subprocess.PIPE,
                                                       stdout=asyncio.subprocess.PIPE,
                                                       stderr=asyncio.subprocess.PIPE)
    except FileNotFoundError as error:
        raise RuntimeError(f"Could not find Node.js command: {config.node_command}") from error

    communicate_task = asyncio.create_task(process.communicate(input=payload))
    done, _pending = await asyncio.wait({communicate_task}, timeout=config.timeout_seconds)
    if not done:
        process.kill()
        stdout_bytes, stderr_bytes = await communicate_task
        stderr = stderr_bytes.decode(errors="replace").strip()
        stdout = stdout_bytes.decode(errors="replace").strip()
        details = "\n".join(part for part in [stderr, stdout] if part)
        if details:
            raise RuntimeError(f"Codex SDK timed out after {config.timeout_seconds} seconds. Recent SDK output: "
                               f"{_clip(details, 4000)}")
        raise RuntimeError(f"Codex SDK timed out after {config.timeout_seconds} seconds")

    stdout_bytes, stderr_bytes = communicate_task.result()

    stdout = stdout_bytes.decode(errors="replace").strip()
    stderr = stderr_bytes.decode(errors="replace").strip()
    if process.returncode:
        details = "\n".join(part for part in [stderr, stdout] if part)
        raise RuntimeError(f"Codex SDK failed with exit code {process.returncode}: {_clip(details, 4000)}")

    try:
        result = json.loads(stdout)
    except json.JSONDecodeError:
        return _clip(stdout, config.max_output_chars)

    text = result.get("text") if isinstance(result, dict) else None
    if not isinstance(text, str):
        text = json.dumps(result, indent=2, sort_keys=True)
    return _clip(text, config.max_output_chars)


@register_function(config_type=CodexAgentWorkflowConfig)
async def codex_agent(config: CodexAgentWorkflowConfig, _builder: Builder):

    async def _response_fn(chat_request_or_message: ChatRequestOrMessage) -> ChatResponse | str:
        message = GlobalTypeConverter.get().convert(chat_request_or_message, to_type=ChatRequestOrMessage)
        prompt = _build_prompt(message, config)
        content = await _run_node_sdk(prompt=prompt, config=config)

        if message.is_string:
            return content
        return _as_response(content, prompt=prompt, model=config.model)

    async def _stream_fn(chat_request_or_message: ChatRequestOrMessage) -> AsyncGenerator[ChatResponseChunk]:
        message = GlobalTypeConverter.get().convert(chat_request_or_message, to_type=ChatRequestOrMessage)
        prompt = _build_prompt(message, config)
        yield ChatResponseChunk.from_string(await _run_node_sdk(prompt=prompt, config=config),
                                            model=config.model or "codex-agent")

    yield FunctionInfo.create(single_fn=_response_fn, stream_fn=_stream_fn, description=config.description)
