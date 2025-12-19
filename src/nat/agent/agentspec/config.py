# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from pydantic import Field, model_validator

from nat.data_models.agent import AgentBaseConfig
from nat.data_models.component_ref import FunctionGroupRef, FunctionRef


class AgentSpecWorkflowConfig(AgentBaseConfig, name="agent_spec"):
    """
    NAT function that executes an Agent Spec configuration via the LangGraph adapter.

    Provide exactly one of agentspec_yaml, agentspec_json, or agentspec_path.
    Optionally supply tool_names to make NAT/LC tools available to the Agent Spec runtime.
    """

    description: str = Field(default="Agent Spec Workflow", description="Description of this workflow.")

    agentspec_yaml: str | None = Field(default=None, description="Inline Agent Spec YAML content")
    agentspec_json: str | None = Field(default=None, description="Inline Agent Spec JSON content")
    agentspec_path: str | None = Field(default=None, description="Path to an Agent Spec YAML/JSON file")

    tool_names: list[FunctionRef | FunctionGroupRef] = Field(
        default_factory=list, description="Optional list of tool names/groups to expose to the Agent Spec runtime."
    )

    max_history: int = Field(default=15, description="Maximum number of messages to keep in conversation history.")

    @model_validator(mode="after")
    def _validate_sources(self):
        provided = [self.agentspec_yaml, self.agentspec_json, self.agentspec_path]
        cnt = sum(1 for v in provided if v)
        if cnt != 1:
            raise ValueError("Exactly one of agentspec_yaml, agentspec_json, or agentspec_path must be provided")
        return self


def read_agentspec_payload(config: AgentSpecWorkflowConfig) -> tuple[str, str]:
    """Return (format, payload_str) where format is 'yaml' or 'json'."""
    if config.agentspec_yaml:
        return ("yaml", config.agentspec_yaml)
    if config.agentspec_json:
        return ("json", config.agentspec_json)
    assert config.agentspec_path
    path = Path(config.agentspec_path)
    text = path.read_text(encoding="utf-8")
    ext = path.suffix.lower()
    fmt = "json" if ext == ".json" else "yaml"
    return (fmt, text)

