# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""End-to-end test for prediction trie workflow."""

import tempfile
from pathlib import Path

from nat.data_models.intermediate_step import IntermediateStep
from nat.data_models.intermediate_step import IntermediateStepPayload
from nat.data_models.intermediate_step import IntermediateStepType
from nat.data_models.intermediate_step import UsageInfo
from nat.data_models.invocation_node import InvocationNode
from nat.data_models.profiler import PredictionTrieConfig
from nat.data_models.profiler import ProfilerConfig
from nat.profiler.callbacks.token_usage_base_model import TokenUsageBaseModel
from nat.profiler.prediction_trie import load_prediction_trie
from nat.profiler.prediction_trie.trie_lookup import PredictionTrieLookup
from nat.profiler.profile_runner import ProfilerRunner


def make_agent_trace(agent_name: str, num_llm_calls: int, base_timestamp: float) -> list[IntermediateStep]:
    """Create a trace with multiple LLM calls in an agent."""
    steps = []
    ts = base_timestamp

    for i in range(num_llm_calls):
        llm_uuid = f"llm-{agent_name}-{i}"

        # LLM_START
        steps.append(
            IntermediateStep(
                parent_id="root",
                function_ancestry=InvocationNode(
                    function_id=f"{agent_name}-1",
                    function_name=agent_name,
                    parent_id="workflow-1",
                    parent_name="my_workflow",
                ),
                payload=IntermediateStepPayload(
                    event_type=IntermediateStepType.LLM_START,
                    event_timestamp=ts,
                    UUID=llm_uuid,
                ),
            ))
        ts += 0.5

        # LLM_END
        completion_tokens = 100 + (i * 50)  # Vary tokens by position
        steps.append(
            IntermediateStep(
                parent_id="root",
                function_ancestry=InvocationNode(
                    function_id=f"{agent_name}-1",
                    function_name=agent_name,
                    parent_id="workflow-1",
                    parent_name="my_workflow",
                ),
                payload=IntermediateStepPayload(
                    event_type=IntermediateStepType.LLM_END,
                    event_timestamp=ts,
                    span_event_timestamp=ts - 0.5,
                    UUID=llm_uuid,
                    usage_info=UsageInfo(token_usage=TokenUsageBaseModel(completion_tokens=completion_tokens)),
                ),
            ))
        ts += 0.5

    return steps


async def test_e2e_prediction_trie_workflow():
    """Test the complete flow: profiler -> trie -> lookup."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)

        # Create multiple traces with different agents
        traces = [
            make_agent_trace("react_agent", num_llm_calls=3, base_timestamp=1000.0),
            make_agent_trace("react_agent", num_llm_calls=3, base_timestamp=2000.0),
            make_agent_trace("tool_agent", num_llm_calls=2, base_timestamp=3000.0),
        ]

        # Run profiler
        config = ProfilerConfig(
            base_metrics=True,
            prediction_trie=PredictionTrieConfig(enable=True),
        )
        runner = ProfilerRunner(config, output_dir)
        await runner.run(traces)

        # Load trie
        trie_path = output_dir / "prediction_trie.json"
        assert trie_path.exists(), "Trie file should exist"

        trie = load_prediction_trie(trie_path)
        lookup = PredictionTrieLookup(trie)

        # Test lookups
        # react_agent has 3 LLM calls, so at call 1 there are 2 remaining
        result = lookup.find(path=["my_workflow", "react_agent"], call_index=1)
        assert result is not None
        assert result.remaining_calls.mean == 2.0  # 2 remaining after first call

        # At call 3 there are 0 remaining
        result = lookup.find(path=["my_workflow", "react_agent"], call_index=3)
        assert result is not None
        assert result.remaining_calls.mean == 0.0

        # tool_agent should have different stats
        result = lookup.find(path=["my_workflow", "tool_agent"], call_index=1)
        assert result is not None
        assert result.remaining_calls.mean == 1.0  # 1 remaining after first call

        # Unknown agent should fall back to aggregated
        result = lookup.find(path=["my_workflow", "unknown_agent"], call_index=1)
        assert result is not None  # Should still get a result from fallback
