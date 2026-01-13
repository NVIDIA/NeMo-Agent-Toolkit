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
        {
            "agentspec_yaml": "a", "agentspec_json": "{}"
        },
        {
            "agentspec_yaml": "a", "agentspec_path": "p"
        },
        {
            "agentspec_json": "{}", "agentspec_path": "p"
        },
        {
            "agentspec_yaml": "a", "agentspec_json": "{}", "agentspec_path": "p"
        },
    ],
)
def test_agentspec_config_validation_errors(kwargs):
    with pytest.raises(ValueError):
        AgentSpecWorkflowConfig(llm_name="dummy", **kwargs)
