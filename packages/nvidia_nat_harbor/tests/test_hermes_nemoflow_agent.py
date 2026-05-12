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
"""Unit tests for the experimental Hermes NeMo-Flow Harbor wrapper."""

from __future__ import annotations

import json
import shlex
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
import yaml
from harbor.agents.installed.hermes import Hermes
from harbor.models.agent.context import AgentContext

from nat_harbor.agents.installed.hermes_nemoflow import HermesNeMoFlow


def _make_agent(tmp_path: Path, **kwargs) -> HermesNeMoFlow:
    logs_dir = tmp_path / "agent"
    logs_dir.mkdir(parents=True)
    return HermesNeMoFlow(
        logs_dir=logs_dir,
        model_name="nvidia/opus-frontier",
        nemo_flow_repo=str(tmp_path / "nemo-flow"),
        **kwargs,
    )


def _write_minimal_nemo_flow_checkout(path: Path) -> None:
    for file_path in (
            path / "Cargo.toml",
            path / "Cargo.lock",
            path / "crates" / "core" / "Cargo.toml",
            path / "crates" / "cli" / "Cargo.toml",
            path / "crates" / "cli" / "src" / "main.rs",
    ):
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text("# test\n", encoding="utf-8")


def test_name_is_hermes_nemoflow() -> None:
    assert HermesNeMoFlow.name() == "hermes-nemoflow"


def test_validate_local_nemo_flow_checkout_requires_cli_crate(tmp_path: Path) -> None:
    agent = _make_agent(tmp_path)

    with pytest.raises(FileNotFoundError, match="crates/cli/Cargo.toml"):
        agent._validate_local_nemo_flow_checkout()


def test_prepare_upload_tree_copies_cli_sources(tmp_path: Path) -> None:
    source = tmp_path / "nemo-flow"
    _write_minimal_nemo_flow_checkout(source)
    agent = _make_agent(tmp_path)

    upload_tree = agent._prepare_upload_tree()

    assert (upload_tree / "Cargo.toml").exists()
    assert (upload_tree / "crates" / "cli" / "src" / "main.rs").exists()


def test_build_config_yaml_includes_gateway_hooks(tmp_path: Path) -> None:
    agent = _make_agent(tmp_path)
    route = agent._build_provider_route("nvidia", "opus-frontier")

    config = yaml.safe_load(agent._build_config_yaml_with_gateway_hooks(route))

    assert config["model"]["provider"] == "custom"
    assert config["model"]["base_url"] == "${NEMO_FLOW_GATEWAY_URL}"
    assert config["model"]["api_key"] == "${OPENAI_API_KEY}"
    hook_command = config["hooks"]["pre_api_request"][0]["command"]
    assert hook_command.startswith("nemo-flow hook-forward hermes")
    assert "--atif-dir /logs/agent/nemo-flow-gateway-atif" in hook_command
    # The PR #89 observability plugin is process-global and belongs on
    # `nemo-flow run`, not on per-hook `hook-forward` calls.
    assert "--atof-dir" not in hook_command
    assert "--plugin-config" not in hook_command
    assert "--sidecar-url" not in hook_command
    assert "--gateway-mode passthrough" in hook_command
    assert "subagent_start" in config["hooks"]


def test_build_run_env_handles_nvidia(tmp_path: Path) -> None:
    agent = _make_agent(
        tmp_path,
        extra_env={
            "NVIDIA_API_KEY": "nvidia-key",
            "NVIDIA_BASE_URL": "https://nvidia.example/v1",
        },
    )
    route = agent._build_provider_route("nvidia", "opus-frontier")

    env = agent._build_run_env(route, "fix it")

    assert env["NVIDIA_API_KEY"] == "nvidia-key"
    assert env["OPENAI_API_KEY"] == "nvidia-key"
    assert env["HARBOR_INSTRUCTION"] == "fix it"


def test_build_run_env_requires_provider_api_key(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    agent = _make_agent(tmp_path)
    route = agent._build_provider_route("nvidia", "opus-frontier")

    with pytest.raises(ValueError, match="NVIDIA_API_KEY"):
        agent._build_run_env(route, "fix it")


@pytest.mark.asyncio
async def test_install_can_use_prebuilt_nemo_flow(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_hermes_install(self: HermesNeMoFlow, environment: AsyncMock) -> None:
        return None

    agent = _make_agent(tmp_path, use_prebuilt_nemo_flow=True)
    monkeypatch.setattr(Hermes, "install", fake_hermes_install)
    monkeypatch.setattr(agent, "_prepare_upload_tree", lambda: pytest.fail("should not upload source"))
    mock_env = AsyncMock()
    mock_env.exec.return_value = AsyncMock(return_code=0, stdout="", stderr="")

    await agent.install(mock_env)

    commands = [call.kwargs["command"] for call in mock_env.exec.call_args_list]
    assert any("command -v nemo-flow" in command for command in commands)
    assert any("nemo-flow run --help" in command for command in commands)
    assert all("cargo build" not in command for command in commands)


@pytest.mark.asyncio
async def test_run_uses_gateway_wrapper(tmp_path: Path) -> None:
    agent = _make_agent(
        tmp_path,
        extra_env={
            "NVIDIA_API_KEY": "nvidia-key",
            "NVIDIA_BASE_URL": "https://nvidia.example/v1",
        },
    )
    mock_env = AsyncMock()
    mock_env.exec.return_value = AsyncMock(return_code=0, stdout="", stderr="")

    await agent.run("solve the task", mock_env, AsyncMock())

    commands = [call.kwargs["command"] for call in mock_env.exec.call_args_list]
    # Use a token boundary to avoid matching "nemo-flow run" as a substring of
    # any other command (defensive against future log/tee changes).
    run_command = next(command for command in commands if " nemo-flow run " in f" {command} ")
    assert "--agent hermes" in run_command
    assert "--atif-dir /logs/agent/nemo-flow-gateway-atif" in run_command
    assert "--atof-dir" not in run_command
    assert "--plugin-config" not in run_command
    assert "--openai-base-url https://nvidia.example/v1" in run_command
    assert "hermes --yolo chat" in run_command
    assert "tee /logs/agent/hermes.txt" in run_command
    assert "on_session_finalize" in run_command
    assert "read_text(encoding=" in run_command
    assert "utf-8" in run_command
    assert "gateway-gateway" in run_command


def test_gateway_run_can_enable_observability_plugin_config(tmp_path: Path) -> None:
    agent = _make_agent(
        tmp_path,
        enable_nemoflow_observability_plugin=True,
        atof_dir="/logs/agent/raw-atof",
        plugin_atif_dir="/logs/agent/plugin-atif",
        extra_env={
            "NVIDIA_API_KEY": "nvidia-key",
            "NVIDIA_BASE_URL": "https://nvidia.example/v1",
        },
    )
    route = agent._build_provider_route("nvidia", "opus-frontier")

    command = agent._build_gateway_run_command(route)

    nemo_flow_segment = command.split("&& ", 1)[1].split(" 2>&1", 1)[0]
    args = shlex.split(nemo_flow_segment)
    plugin_index = args.index("--plugin-config")
    plugin_config = json.loads(args[plugin_index + 1])
    observability = plugin_config["components"][0]
    assert observability["kind"] == "observability"
    assert observability["config"]["atof"] == {
        "enabled": True,
        "output_directory": "/logs/agent/raw-atof",
        "filename": "events.jsonl",
        "mode": "overwrite",
    }
    assert observability["config"]["atif"] == {
        "enabled": True,
        "output_directory": "/logs/agent/plugin-atif",
        "filename_template": "trajectory-{session_id}.atif.json",
    }


def test_populate_context_converts_atof_and_sets_tokens(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    agent = _make_agent(tmp_path)
    atof_dir = agent.logs_dir / "nemo-flow-atof"
    atof_dir.mkdir(parents=True)
    atof_path = atof_dir / "events.jsonl"
    atof_path.write_text('{"event_type":"mark"}\n', encoding="utf-8")

    gateway_dir = agent.logs_dir / "nemo-flow-gateway-atif"
    gateway_dir.mkdir(parents=True)
    source = gateway_dir / "session-1.atif.json"
    source.write_text(
        json.dumps({
            "schema_version": "ATIF-v1.6",
            "final_metrics": {
                "total_prompt_tokens": 12,
                "total_completion_tokens": 7,
            },
            "steps": [],
        }),
        encoding="utf-8",
    )

    def fake_convert_atof_to_atif(input_path: Path, output_path: Path) -> Path:
        assert input_path == atof_path
        output_path.parent.mkdir(parents=True)
        output_path.write_text(
            json.dumps({
                "schema_version": "ATIF-v1.7",
                "final_metrics": {
                    "total_prompt_tokens": 21,
                    "total_completion_tokens": 13,
                },
                "steps": [],
            }),
            encoding="utf-8",
        )
        return output_path

    monkeypatch.setattr(agent, "_convert_atof_to_atif", fake_convert_atof_to_atif)
    context = AgentContext()

    agent.populate_context_post_run(context)

    canonical = gateway_dir / "trajectory.json"
    converted = agent.logs_dir / "nemo-flow-atof-atif" / "trajectory.json"
    assert canonical.exists()
    assert converted.exists()
    assert context.n_input_tokens == 21
    assert context.n_output_tokens == 13
    assert context.metadata is not None
    assert context.metadata["nemo_flow_instrumentation"] == "gateway"
    assert context.metadata["nemo_flow_observability_plugin_enabled"] is False
    assert context.metadata["nemo_flow_atof_path"] == str(atof_path)
    assert context.metadata["nemo_flow_converted_atif_path"] == str(converted)
    assert context.metadata["nemo_flow_gateway_canonical_atif_path"] == str(canonical)
    assert context.metadata["nemo_flow_plugin_canonical_atif_exists"] is False


def test_populate_context_can_use_plugin_atif_when_atof_is_absent(tmp_path: Path) -> None:
    agent = _make_agent(tmp_path, enable_nemoflow_observability_plugin=True)
    plugin_dir = agent.logs_dir / "nemo-flow-plugin-atif"
    plugin_dir.mkdir(parents=True)
    source = plugin_dir / "trajectory-session-1.atif.json"
    source.write_text(
        json.dumps({
            "schema_version": "ATIF-v1.6",
            "final_metrics": {
                "total_prompt_tokens": 34,
                "total_completion_tokens": 21,
            },
            "steps": [],
        }),
        encoding="utf-8",
    )
    gateway_dir = agent.logs_dir / "nemo-flow-gateway-atif"
    gateway_dir.mkdir(parents=True)
    (gateway_dir / "session-1.atif.json").write_text(
        json.dumps({
            "schema_version": "ATIF-v1.6",
            "final_metrics": {
                "total_prompt_tokens": 1,
                "total_completion_tokens": 1,
            },
            "steps": [],
        }),
        encoding="utf-8",
    )

    context = AgentContext()
    agent.populate_context_post_run(context)

    plugin_canonical = plugin_dir / "trajectory.json"
    assert plugin_canonical.exists()
    assert context.n_input_tokens == 34
    assert context.n_output_tokens == 21
    assert context.metadata is not None
    assert context.metadata["nemo_flow_observability_plugin_enabled"] is True
    assert context.metadata["nemo_flow_plugin_canonical_atif_path"] == str(plugin_canonical)
    assert context.metadata["nemo_flow_plugin_canonical_atif_exists"] is True


def test_populate_context_default_does_not_raise_when_atof_missing(tmp_path: Path) -> None:
    """ATOF is best-effort/optional until Hermes gateway events are wired into PR #88."""
    agent = _make_agent(tmp_path)
    gateway_dir = agent.logs_dir / "nemo-flow-gateway-atif"
    gateway_dir.mkdir(parents=True)
    (gateway_dir / "session-1.atif.json").write_text(
        json.dumps({
            "schema_version": "ATIF-v1.6", "final_metrics": {}, "steps": []
        }),
        encoding="utf-8",
    )

    context = AgentContext()
    agent.populate_context_post_run(context)

    assert context.metadata is not None
    assert context.metadata["nemo_flow_atof_exists"] is False
    assert context.metadata["nemo_flow_gateway_canonical_atif_exists"] is True


def test_populate_context_raises_when_atof_required_and_missing(tmp_path: Path) -> None:
    """Opt-in fail-on-missing remains supported for branches that emit Hermes ATOF."""
    agent = _make_agent(tmp_path, fail_missing_nemoflow_atof=True)

    with pytest.raises(FileNotFoundError, match="Missing NeMo-Flow ATOF JSONL"):
        agent.populate_context_post_run(AgentContext())


def test_populate_context_raises_when_gateway_atif_missing_by_default(tmp_path: Path) -> None:
    """Gateway ATIF is the canonical artifact today; missing is fatal by default."""
    agent = _make_agent(tmp_path)

    with pytest.raises(FileNotFoundError, match="Missing NeMo-Flow gateway ATIF"):
        agent.populate_context_post_run(AgentContext())


def test_sidecar_atif_dir_kwarg_back_compat(tmp_path: Path) -> None:
    """Older callers passing sidecar_atif_dir= still work; new name takes precedence."""
    agent = _make_agent(tmp_path / "legacy_only", sidecar_atif_dir="/logs/agent/legacy")
    assert agent._gateway_atif_dir == "/logs/agent/legacy"
    assert agent._sidecar_atif_dir == "/logs/agent/legacy"

    agent2 = _make_agent(
        tmp_path / "both",
        sidecar_atif_dir="/logs/agent/legacy",
        gateway_atif_dir="/logs/agent/new",
    )
    assert agent2._gateway_atif_dir == "/logs/agent/new"
