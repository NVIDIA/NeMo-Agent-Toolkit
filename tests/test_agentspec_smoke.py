# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import sys
import pytest

# Prepend local src to sys.path to ensure local NAT is used
SRC_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)


@pytest.mark.asyncio
async def test_agentspec_adapter_smoke(monkeypatch):
    # Minimal Agent Spec defining a simple agent that just echoes user input
    spec_yaml = """
component_type: Agent
name: echo-agent
description: echo
"""

    # Monkeypatch adapter to return a stub runnable that returns the input
    class StubComponent:
        async def ainvoke(self, value):
            if isinstance(value, dict) and "messages" in value:
                msgs = value["messages"]
                last_user = next((m.get("content") for m in reversed(msgs) if m.get("role") == "user"), "")
                return {"output": last_user}
            return {"output": str(value)}

    class StubLoader:
        def __init__(self, *args, **kwargs):
            pass

        def load_yaml(self, _):
            return StubComponent()

    import types

    fake_mod = types.ModuleType("langgraph_agentspec_adapter.agentspecloader")
    fake_mod.AgentSpecLoader = StubLoader
    sys.modules["langgraph_agentspec_adapter"] = types.ModuleType("langgraph_agentspec_adapter")
    sys.modules["langgraph_agentspec_adapter.agentspecloader"] = fake_mod

    import nat.agent.agentspec.register  # noqa: F401 ensure registration
    from nat.agent.agentspec.config import AgentSpecWorkflowConfig
    from nat.builder.workflow_builder import WorkflowBuilder

    cfg = AgentSpecWorkflowConfig(llm_name="dummy", agentspec_yaml=spec_yaml)
    async with WorkflowBuilder() as builder:
        fn = await builder.set_workflow(cfg)
        out = await fn.acall_invoke(input_message="hello world")
        assert out is not None
