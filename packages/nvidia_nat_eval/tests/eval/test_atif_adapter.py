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

from nat.data_models.atif import ATIFAgentConfig
from nat.data_models.atif import ATIFTrajectory
from nat.data_models.evaluator import EvalInput
from nat.data_models.evaluator import EvalInputItem
from nat.data_models.intermediate_step import IntermediateStep
from nat.data_models.intermediate_step import IntermediateStepPayload
from nat.data_models.intermediate_step import IntermediateStepType
from nat.data_models.intermediate_step import InvocationNode
from nat.data_models.intermediate_step import StreamEventData
from nat.plugins.eval.runtime.atif_adapter import EvalAtifAdapter


def _make_eval_input_item(item_id: str = "item-1") -> EvalInputItem:
    step = IntermediateStep(parent_id="root",
                            function_ancestry=InvocationNode(function_name="llm_test", function_id="llm-test"),
                            payload=IntermediateStepPayload(event_type=IntermediateStepType.LLM_END,
                                                            data=StreamEventData(input="input", output="output")))
    return EvalInputItem(id=item_id,
                         input_obj="input",
                         expected_output_obj="expected",
                         output_obj="actual",
                         trajectory=[step],
                         full_dataset_entry={"id": item_id})


class _CountingConverter:

    def __init__(self) -> None:
        self.calls = 0

    def convert(self, steps: list[IntermediateStep], *, session_id: str | None = None, agent_name: str | None = None):
        self.calls += 1
        return ATIFTrajectory(session_id=session_id or "sid",
                              agent=ATIFAgentConfig(name=agent_name or "nat-agent", version="0.0.0"))


def test_private_ensure_cache_converts_once_per_item():
    converter = _CountingConverter()
    adapter = EvalAtifAdapter(converter=converter)
    eval_input = EvalInput(eval_input_items=[_make_eval_input_item("1")])

    adapter._ensure_cache(eval_input)
    adapter._ensure_cache(eval_input)

    assert converter.calls == 1


def test_build_samples_uses_prebuilt_trajectory_without_conversion():
    converter = _CountingConverter()
    adapter = EvalAtifAdapter(converter=converter)
    item = _make_eval_input_item("sample-a")
    eval_input = EvalInput(eval_input_items=[item])
    prebuilt = ATIFTrajectory(session_id="sample-a", agent=ATIFAgentConfig(name="prebuilt-agent", version="0.0.0"))

    samples = adapter.build_samples(eval_input, prebuilt_trajectories={"sample-a": prebuilt})

    assert converter.calls == 0
    assert len(samples) == 1
    assert samples[0].trajectory.agent.name == "prebuilt-agent"
    assert samples[0].item_id == "sample-a"
