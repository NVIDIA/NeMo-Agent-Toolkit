# SPDX-FileCopyrightText: Copyright (c) <year>, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import sys
import types


async def main():
    # Ensure NAT src is importable if running from repo root
    import os
    repo_root = os.path.dirname(os.path.abspath(__file__))
    src_dir = os.path.join(os.path.dirname(repo_root), "src")
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)

    # Force registration imports
    import nat.agent.register  # noqa: F401

    # Create a fake adapter module that returns a stub component
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

    fake_mod = types.ModuleType("langgraph_agentspec_adapter.agentspecloader")
    fake_mod.AgentSpecLoader = StubLoader
    sys.modules["langgraph_agentspec_adapter"] = types.ModuleType("langgraph_agentspec_adapter")
    sys.modules["langgraph_agentspec_adapter.agentspecloader"] = fake_mod

    # Import registers agent workflows (including Agent Spec)
    import nat.agent.agentspec.register  # noqa: F401
    from nat.agent.agentspec.config import AgentSpecWorkflowConfig
    from nat.builder.workflow_builder import WorkflowBuilder

    spec_yaml = """
component_type: Agent
name: echo-agent
description: echo
"""

    cfg = AgentSpecWorkflowConfig(llm_name="dummy", agentspec_yaml=spec_yaml, tool_names=[])
    async with WorkflowBuilder() as builder:
        fn = await builder.set_workflow(cfg)
        out = await fn.acall_invoke(input_message="hello agentspec")
        print("OK:", out)


if __name__ == "__main__":
    asyncio.run(main())
