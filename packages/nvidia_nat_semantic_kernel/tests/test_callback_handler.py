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

import pytest

from nat.utils.reactive.subject import Subject


@pytest.mark.slow
async def test_semantic_kernel_handler_tool_call(reactive_stream: Subject):
    """
    Test that the SK callback logs tool usage events.
    """
    from uuid import uuid4

    from nat.builder.context import Context
    from nat.builder.framework_enum import LLMFrameworkEnum
    from nat.data_models.intermediate_step import IntermediateStepPayload
    from nat.data_models.intermediate_step import IntermediateStepType
    from nat.data_models.intermediate_step import TraceMetadata
    from nat.plugins.semantic_kernel.callback_handler import SemanticKernelProfilerHandler

    all_ = []
    _ = SemanticKernelProfilerHandler(workflow_llms={})
    _ = reactive_stream.subscribe(all_.append)
    step_manager = Context.get().intermediate_step_manager
    # We'll manually simulate the relevant methods.

    # Suppose we do a tool "invoke_function_call"
    # We'll simulate a call to handler's patched function
    run_id1 = str(uuid4())
    start_event = IntermediateStepPayload(UUID=run_id1,
                                          event_type=IntermediateStepType.TOOL_START,
                                          framework=LLMFrameworkEnum.SEMANTIC_KERNEL,
                                          metadata=TraceMetadata(tool_inputs={"args": ["some input"]}))
    step_manager.push_intermediate_step(start_event)

    end_event = IntermediateStepPayload(UUID=run_id1,
                                        event_type=IntermediateStepType.TOOL_END,
                                        framework=LLMFrameworkEnum.SEMANTIC_KERNEL,
                                        metadata=TraceMetadata(tool_outputs={"result": "some result"}))
    step_manager.push_intermediate_step(end_event)

    assert len(all_) == 2
    assert all_[0].event_type == IntermediateStepType.TOOL_START
    assert all_[1].event_type == IntermediateStepType.TOOL_END
    assert all_[1].payload.metadata.tool_outputs == {"result": "some result"}
