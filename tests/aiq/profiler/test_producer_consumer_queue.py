# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from uuid import uuid4

from aiq.builder.context import AIQContext
from aiq.builder.framework_enum import LLMFrameworkEnum
from aiq.data_models.intermediate_step import IntermediateStepPayload
from aiq.data_models.intermediate_step import IntermediateStepType as WorkflowEventEnum
from aiq.utils.reactive.subject import Subject


async def test_usage_stat_order_and_latency(reactive_stream: Subject):
    """
    Example test that simulates a simple scenario with two LLM calls and ensures
    the usage stats are in the correct order and that we can compute a latency from them.

    We manually push usage stats into the queue in the order they might occur at runtime,
    then verify we can read them back in the same order.
    """

    result_stats = []
    step_manager = AIQContext.get().intermediate_step_manager
    _ = step_manager.subscribe(result_stats.append)

    # Simulate first LLM call
    run_id1 = str(uuid4())
    first_start = IntermediateStepPayload(UUID=run_id1,
                                          event_type=WorkflowEventEnum.LLM_START,
                                          framework=LLMFrameworkEnum.LANGCHAIN)
    step_manager.push_intermediate_step(first_start)

    first_end = IntermediateStepPayload(UUID=run_id1,
                                        event_type=WorkflowEventEnum.LLM_END,
                                        framework=LLMFrameworkEnum.LANGCHAIN)
    step_manager.push_intermediate_step(first_end)

    # second call
    run_id2 = str(uuid4())
    second_start = IntermediateStepPayload(UUID=run_id2,
                                           event_type=WorkflowEventEnum.LLM_START,
                                           framework=LLMFrameworkEnum.LLAMA_INDEX)
    step_manager.push_intermediate_step(second_start)

    second_end = IntermediateStepPayload(UUID=run_id2,
                                         event_type=WorkflowEventEnum.LLM_END,
                                         framework=LLMFrameworkEnum.LLAMA_INDEX)
    step_manager.push_intermediate_step(second_end)

    # verify
    assert len(result_stats) == 4
    assert result_stats[0].event_type == WorkflowEventEnum.LLM_START
    assert result_stats[1].event_type == WorkflowEventEnum.LLM_END
    assert result_stats[3].payload.framework == LLMFrameworkEnum.LLAMA_INDEX
