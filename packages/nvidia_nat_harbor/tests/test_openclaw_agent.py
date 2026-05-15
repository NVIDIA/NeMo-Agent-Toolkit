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
"""Unit tests for the OpenClaw Harbor agent vendored under ``nat_harbor``."""

import json
from pathlib import Path

import pytest
from harbor.agents.factory import AgentFactory
from harbor.models.agent.context import AgentContext
from harbor.models.agent.name import AgentName
from harbor.models.trial.config import AgentConfig

from nat_harbor.agents.installed.openclaw import OPENCLAW_AGENT_SETUP_TIMEOUT_SEC
from nat_harbor.agents.installed.openclaw import OpenClaw
from nat_harbor.agents.installed.openclaw import openclaw_session_jsonl_to_atif_steps

_HARBOR_ENUM_HAS_OPENCLAW = hasattr(AgentName, "OPENCLAW")

skip_if_harbor_missing_openclaw_enum = pytest.mark.skipif(
    not _HARBOR_ENUM_HAS_OPENCLAW,
    reason=(
        "Harbor ``AgentName`` has no ``OPENCLAW``; use a Harbor revision that "
        "registers OpenClaw or exercise the agent via ``--agent-import-path`` only."
    ),
)


@pytest.fixture
def agent(tmp_path: Path) -> OpenClaw:
    return OpenClaw(
        logs_dir=tmp_path,
        model_name="anthropic/claude-sonnet-4-20250514",
    )


@skip_if_harbor_missing_openclaw_enum
def test_name(agent: OpenClaw) -> None:
    assert agent.name() == AgentName.OPENCLAW.value


def test_load_json_object_trailing_noise(agent: OpenClaw) -> None:
    raw = 'prefix noise\n{"payloads": [], "meta": {}}\n'
    parsed = agent._load_json_object(raw)
    assert parsed == {"payloads": [], "meta": {}}


def test_load_json_object_stale_brace_before_envelope(agent: OpenClaw) -> None:
    raw = (
        '[tools] raw_params={"path": "/x"}\n'
        '{"payloads": [{"text": "ok"}], "meta": {"agentMeta": {"sessionId": "s"}}}\n'
    )
    parsed = agent._load_json_object(raw)
    assert parsed is not None
    assert parsed["meta"]["agentMeta"]["sessionId"] == "s"


def test_convert_envelope_basic(agent: OpenClaw) -> None:
    envelope = {
        "payloads": [
            {"text": "hello", "isReasoning": False},
            {"text": "think", "isReasoning": True},
        ],
        "meta": {
            "agentMeta": {
                "sessionId": "sess-abc",
                "usage": {"input": 10, "output": 5, "cacheRead": 2},
            },
        },
    }
    traj = agent._convert_envelope_to_trajectory(envelope, "do the thing")
    assert traj is not None
    assert traj.session_id == "sess-abc"
    assert len(traj.steps) == 2
    assert traj.steps[0].source == "user"
    assert traj.steps[0].message == "do the thing"
    assert traj.steps[1].source == "agent"
    assert traj.steps[1].message == "hello"
    assert traj.steps[1].reasoning_content == "think"
    assert traj.final_metrics is not None
    assert traj.final_metrics.total_prompt_tokens == 12
    assert traj.final_metrics.total_completion_tokens == 5
    assert traj.final_metrics.total_cached_tokens == 2


def test_populate_context_writes_trajectory(agent: OpenClaw) -> None:
    payload = {
        "payloads": [{"text": "ok"}],
        "meta": {"agentMeta": {"sessionId": "s1", "usage": {}}},
    }
    (agent.logs_dir / "openclaw.txt").write_text(json.dumps(payload, indent=2))
    (agent.logs_dir / "instruction.txt").write_text("task text")

    ctx = AgentContext()
    agent.populate_context_post_run(ctx)

    traj_path = agent.logs_dir / "trajectory.json"
    assert traj_path.is_file()
    out = json.loads(traj_path.read_text())
    assert out["session_id"] == "s1"
    assert len(out["steps"]) == 2
    assert out["steps"][0]["message"] == "task text"


def test_compose_config_patch_mcp(agent: OpenClaw, tmp_path: Path) -> None:
    from harbor.models.task.config import MCPServerConfig

    a = OpenClaw(
        logs_dir=tmp_path,
        model_name="openai/gpt-4.1",
        mcp_servers=[
            MCPServerConfig(
                name="demo",
                transport="stdio",
                command="mcp",
                args=["--stdio"],
            ),
        ],
        openclaw_config={"agents": {"defaults": {"verboseDefault": "off"}}},
    )
    cfg = a._build_full_openclaw_config()
    assert cfg["agents"]["defaults"]["verboseDefault"] == "off"
    assert cfg["mcp"]["servers"]["demo"]["command"] == "mcp"
    assert cfg["mcp"]["servers"]["demo"]["args"] == ["--stdio"]


def test_failover_retries_kwarg_overrides_openclaw_config(tmp_path: Path) -> None:
    a = OpenClaw(
        logs_dir=tmp_path,
        model_name="openai/gpt-4.1",
        failover_retries=7,
        openclaw_config={"auth": {"cooldowns": {"rateLimitedProfileRotations": 1}}},
    )
    cfg = a._build_full_openclaw_config()
    assert cfg["auth"]["cooldowns"]["rateLimitedProfileRotations"] == 7


def test_failover_retries_kwarg_sets_auth_cooldowns(tmp_path: Path) -> None:
    a = OpenClaw(
        logs_dir=tmp_path,
        model_name="openai/gpt-4.1",
        failover_retries=4,
    )
    cfg = a._build_full_openclaw_config()
    assert cfg["auth"]["cooldowns"]["rateLimitedProfileRotations"] == 4


def test_nvidia_base_url_from_env_in_uploaded_config(tmp_path: Path) -> None:
    inference = "https://inference-api.nvidia.com/v1"
    a = OpenClaw(
        logs_dir=tmp_path,
        model_name="nvidia/opus-frontier",
        extra_env={"NVIDIA_BASE_URL": inference},
    )
    cfg = a._build_full_openclaw_config()
    assert cfg["models"]["providers"]["nvidia"]["baseUrl"] == inference
    nvidia_models = cfg["models"]["providers"]["nvidia"]["models"]
    assert isinstance(nvidia_models, list)
    assert len(nvidia_models) == 1
    assert nvidia_models[0]["id"] == "nvidia/opus-frontier"


def test_nvidia_provider_baseurl_only_gets_models_array(tmp_path: Path) -> None:
    """User YAML may set only ``baseUrl``; OpenClaw requires a ``models`` array."""
    custom = "https://example.com/v1"
    a = OpenClaw(
        logs_dir=tmp_path,
        model_name="nvidia/nemotron-3-nano-30b-a3b",
        openclaw_config={
            "models": {"providers": {"nvidia": {"baseUrl": custom}}},
        },
    )
    cfg = a._build_full_openclaw_config()
    assert cfg["models"]["providers"]["nvidia"]["baseUrl"] == custom
    assert isinstance(cfg["models"]["providers"]["nvidia"]["models"], list)
    assert len(cfg["models"]["providers"]["nvidia"]["models"]) == 1
    assert cfg["models"]["providers"]["nvidia"]["models"][0]["id"] == "nvidia/nemotron-3-nano-30b-a3b"


@skip_if_harbor_missing_openclaw_enum
def test_factory_openclaw_default_install_timeout_when_override_unset(
    tmp_path: Path,
) -> None:
    cfg = AgentConfig(name=AgentName.OPENCLAW.value, model_name="openai/gpt-4.1")
    assert cfg.override_setup_timeout_sec is None
    agent = AgentFactory.create_agent_from_config(cfg, logs_dir=tmp_path)
    assert type(agent).__name__ == "OpenClaw"
    assert cfg.override_setup_timeout_sec is None
    assert agent._install_exec_timeout_sec == int(OPENCLAW_AGENT_SETUP_TIMEOUT_SEC)


@skip_if_harbor_missing_openclaw_enum
def test_factory_leaves_explicit_setup_timeout_unchanged(tmp_path: Path) -> None:
    cfg = AgentConfig(
        name=AgentName.OPENCLAW.value,
        model_name="openai/gpt-4.1",
        override_setup_timeout_sec=123.0,
    )
    AgentFactory.create_agent_from_config(cfg, logs_dir=tmp_path)
    assert cfg.override_setup_timeout_sec == 123.0


def test_nemo_flow_plugin_merged_into_openclaw_config(tmp_path: Path) -> None:
    a = OpenClaw(
        logs_dir=tmp_path,
        model_name="openai/gpt-4.1",
    )
    cfg = a._build_full_openclaw_config()
    plugins = cfg["plugins"]
    assert "nemo-flow" in plugins["allow"]
    entry = plugins["entries"]["nemo-flow"]
    assert entry["enabled"] is True
    assert entry["hooks"]["allowConversationAccess"] is True
    assert cfg["plugins"]["bundledDiscovery"] == "compat"
    assert entry["config"]["enabled"] is True
    assert entry["config"]["backend"] == "hooks"
    comps = entry["config"]["plugins"]["components"]
    obs = next(c for c in comps if c["kind"] == "observability")
    assert obs["config"]["atif"]["enabled"] is True
    assert obs["config"]["atif"]["output_directory"] == "/logs/agent/nemo-flow-atif"
    assert obs["config"]["opentelemetry"]["enabled"] is False
    assert obs["config"]["openinference"]["enabled"] is False


def test_nemo_flow_build_skips_harbor_plugin_merge_when_flag_false(tmp_path: Path) -> None:
    a = OpenClaw(
        logs_dir=tmp_path,
        model_name="openai/gpt-4.1",
    )
    cfg = a._build_full_openclaw_config(include_nemo_flow_plugin=False)
    assert "plugins" not in cfg
    cfg_full = a._build_full_openclaw_config()
    assert "nemo-flow" in cfg_full["plugins"]["allow"]


def test_nemo_flow_merge_appends_allow_preserves_other_plugins(tmp_path: Path) -> None:
    a = OpenClaw(
        logs_dir=tmp_path,
        model_name="openai/gpt-4.1",
        openclaw_config={
            "plugins": {
                "allow": ["custom-plugin"],
                "entries": {"custom-plugin": {"enabled": True}},
            },
        },
    )
    cfg = a._build_full_openclaw_config()
    assert cfg["plugins"]["allow"] == ["custom-plugin", "nemo-flow"]
    assert "custom-plugin" in cfg["plugins"]["entries"]
    assert cfg["plugins"]["entries"]["nemo-flow"]["enabled"] is True


def test_nemo_flow_disabled_skips_plugin_merge(tmp_path: Path) -> None:
    a = OpenClaw(
        logs_dir=tmp_path,
        model_name="openai/gpt-4.1",
        enable_nemo_flow=False,
    )
    cfg = a._build_full_openclaw_config()
    assert "plugins" not in cfg


def test_nemo_flow_user_can_disable_plugin_entry(tmp_path: Path) -> None:
    a = OpenClaw(
        logs_dir=tmp_path,
        model_name="openai/gpt-4.1",
        openclaw_config={
            "plugins": {
                "entries": {
                    "nemo-flow": {
                        "enabled": False,
                    },
                },
            },
        },
    )
    cfg = a._build_full_openclaw_config()
    assert cfg["plugins"]["entries"]["nemo-flow"]["enabled"] is False
    assert "nemo-flow" in cfg["plugins"]["allow"]


def test_nvidia_base_url_openclaw_config_wins(tmp_path: Path) -> None:
    custom = "https://example.com/v1"
    a = OpenClaw(
        logs_dir=tmp_path,
        model_name="nvidia/opus-frontier",
        extra_env={"NVIDIA_BASE_URL": "https://inference-api.nvidia.com/v1"},
        openclaw_config={
            "models": {"providers": {"nvidia": {"baseUrl": custom}}},
        },
    )
    cfg = a._build_full_openclaw_config()
    assert cfg["models"]["providers"]["nvidia"]["baseUrl"] == custom
    nvidia_models = cfg["models"]["providers"]["nvidia"]["models"]
    assert isinstance(nvidia_models, list)
    assert len(nvidia_models) == 1
    assert nvidia_models[0]["id"] == "nvidia/opus-frontier"


def test_openclaw_session_jsonl_to_atif_steps_minimal(tmp_path: Path) -> None:
    session = tmp_path / "openclaw.session.jsonl"
    session.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "type": "message",
                        "timestamp": "2026-01-01T00:00:00Z",
                        "message": {
                            "role": "user",
                            "content": [{"type": "text", "text": "hi"}],
                        },
                    }
                ),
                json.dumps(
                    {
                        "type": "message",
                        "timestamp": "2026-01-01T00:00:01Z",
                        "message": {
                            "role": "assistant",
                            "content": [
                                {"type": "text", "text": "hello "},
                                {
                                    "type": "toolCall",
                                    "id": "c1",
                                    "name": "exec",
                                    "arguments": {"command": "x"},
                                },
                            ],
                            "usage": {"input": 1, "output": 2, "cacheRead": 0},
                        },
                    }
                ),
                json.dumps(
                    {
                        "type": "message",
                        "timestamp": "2026-01-01T00:00:02Z",
                        "message": {
                            "role": "toolResult",
                            "toolCallId": "c1",
                            "toolName": "exec",
                            "content": [{"type": "text", "text": "out"}],
                            "details": {"aggregated": "out"},
                        },
                    }
                ),
                json.dumps(
                    {
                        "type": "message",
                        "timestamp": "2026-01-01T00:00:03Z",
                        "message": {
                            "role": "assistant",
                            "content": [{"type": "text", "text": "done"}],
                            "usage": {"input": 3, "output": 4, "cacheRead": 0},
                        },
                    }
                ),
            ]
        )
        + "\n"
    )
    steps = openclaw_session_jsonl_to_atif_steps(
        session,
        instruction="task from instruction",
        model_name="anthropic/claude-sonnet-4-20250514",
    )
    assert steps is not None
    assert len(steps) == 3
    assert steps[0].message == "task from instruction"
    assert steps[1].tool_calls is not None
    assert steps[1].observation is not None


def test_populate_context_optional_session_jsonl(tmp_path: Path) -> None:
    session = tmp_path / "openclaw.session.jsonl"
    session.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "type": "message",
                        "message": {
                            "role": "user",
                            "content": [{"type": "text", "text": "u"}],
                        },
                    }
                ),
                json.dumps(
                    {
                        "type": "message",
                        "message": {
                            "role": "assistant",
                            "content": [{"type": "text", "text": "a"}],
                            "usage": {"input": 1, "output": 1, "cacheRead": 0},
                        },
                    }
                ),
            ]
        )
        + "\n"
    )
    payload = {
        "payloads": [{"text": "summary"}],
        "meta": {"agentMeta": {"sessionId": "s1", "usage": {"input": 9, "output": 9}}},
    }
    agent = OpenClaw(
        logs_dir=tmp_path,
        model_name="openai/gpt-4.1",
        use_openclaw_session_jsonl_for_steps=True,
    )
    (tmp_path / "openclaw.txt").write_text(json.dumps(payload))
    (tmp_path / "instruction.txt").write_text("instr")
    ctx = AgentContext()
    agent.populate_context_post_run(ctx)
    out = json.loads((tmp_path / "trajectory.json").read_text())
    assert len(out["steps"]) == 2
    assert out["steps"][1]["message"] == "a"
