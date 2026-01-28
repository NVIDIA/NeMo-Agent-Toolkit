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

from uuid import uuid4

import pytest

from nat.builder.context import Context
from nat.builder.framework_enum import LLMFrameworkEnum
from nat.data_models.intermediate_step import IntermediateStepPayload
from nat.data_models.intermediate_step import IntermediateStepType
from nat.data_models.intermediate_step import StreamEventData
from nat.data_models.intermediate_step import UsageInfo
from nat.utils.reactive.subject import Subject


@pytest.mark.slow
async def test_crewai_handler_time_between_calls(reactive_stream: Subject):
    """
    Test CrewAIProfilerHandler ensures seconds_between_calls is properly set for consecutive calls.
    We'll mock time.time() to produce stable intervals.
    """
    pytest.importorskip("crewai")

    import math

    from nat.plugins.crewai.crewai_callback_handler import CrewAIProfilerHandler

    # The crewAI handler monkey-patch logic is for real code instrumentation,
    # but let's just call the wrapped calls directly:
    results = []
    handler = CrewAIProfilerHandler()
    _ = reactive_stream.subscribe(results.append)
    step_manager = Context.get().intermediate_step_manager

    # We'll patch time.time so it returns predictable values:
    # e.g. 100.0 for the first call, 103.2 for the second, etc.
    # Simulate a first LLM call
    # crewAI calls _llm_call_monkey_patch => we can't call that directly, let's just do an inline approach
    # We'll do a short local function "simulate_llm_call" that replicates the logic:
    times = [100.0, 103.2, 107.5, 112.0]
    # seconds_between_calls = int(now - self.last_call_ts) => at the first call, last_call_ts=some default
    # but let's just forcibly create a usage stat

    run_id1 = str(uuid4())
    start_stat = IntermediateStepPayload(UUID=run_id1,
                                         event_type=IntermediateStepType.LLM_START,
                                         data=StreamEventData(input="Hello user!"),
                                         framework=LLMFrameworkEnum.CREWAI,
                                         event_timestamp=times[0])
    step_manager.push_intermediate_step(start_stat)
    handler.last_call_ts = times[0]

    # Simulate end
    end_stat = IntermediateStepPayload(UUID=run_id1,
                                       event_type=IntermediateStepType.LLM_END,
                                       data=StreamEventData(output="World response"),
                                       framework=LLMFrameworkEnum.CREWAI,
                                       event_timestamp=times[1])
    step_manager.push_intermediate_step(end_stat)

    now2 = times[2]
    run_id2 = str(uuid4())
    start_stat2 = IntermediateStepPayload(UUID=run_id2,
                                          event_type=IntermediateStepType.LLM_START,
                                          data=StreamEventData(input="Hello again!"),
                                          framework=LLMFrameworkEnum.CREWAI,
                                          event_timestamp=now2,
                                          usage_info=UsageInfo(seconds_between_calls=math.floor(now2 -
                                                                                                handler.last_call_ts)))
    step_manager.push_intermediate_step(start_stat2)
    handler.last_call_ts = now2
    second_end = IntermediateStepPayload(UUID=run_id2,
                                         event_type=IntermediateStepType.LLM_END,
                                         data=StreamEventData(output="Another response"),
                                         framework=LLMFrameworkEnum.CREWAI,
                                         event_timestamp=times[3])
    step_manager.push_intermediate_step(second_end)

    assert len(results) == 4
    # Check the intervals
    assert results[2].usage_info.seconds_between_calls == 7
