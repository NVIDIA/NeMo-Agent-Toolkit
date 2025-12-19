# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

import pytest

from nat.agent.agentspec.config import AgentSpecWorkflowConfig


def test_agentspec_config_exactly_one_source_yaml():
    cfg = AgentSpecWorkflowConfig(llm_name="dummy", agentspec_yaml="component_type: Agent\nname: test")
    assert cfg.agentspec_yaml and not cfg.agentspec_json and not cfg.agentspec_path


def test_agentspec_config_exactly_one_source_json():
    cfg = AgentSpecWorkflowConfig(llm_name="dummy", agentspec_json="{}")
    assert cfg.agentspec_json and not cfg.agentspec_yaml and not cfg.agentspec_path


def test_agentspec_config_exactly_one_source_path(tmp_path):
    p = tmp_path / "spec.yaml"
    p.write_text("component_type: Agent\nname: test", encoding="utf-8")
    cfg = AgentSpecWorkflowConfig(llm_name="dummy", agentspec_path=str(p))
    assert cfg.agentspec_path and not cfg.agentspec_yaml and not cfg.agentspec_json


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"agentspec_yaml": "a", "agentspec_json": "{}"},
        {"agentspec_yaml": "a", "agentspec_path": "p"},
        {"agentspec_json": "{}", "agentspec_path": "p"},
        {"agentspec_yaml": "a", "agentspec_json": "{}", "agentspec_path": "p"},
    ],
)
def test_agentspec_config_validation_errors(kwargs):
    with pytest.raises(ValueError):
        AgentSpecWorkflowConfig(llm_name="dummy", **kwargs)
