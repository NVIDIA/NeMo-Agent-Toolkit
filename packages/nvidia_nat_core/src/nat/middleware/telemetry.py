# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Paired intermediate steps for middleware execution."""

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

from nat.builder.context import Context
from nat.data_models.intermediate_step import IntermediateStepPayload
from nat.data_models.intermediate_step import IntermediateStepType
from nat.data_models.intermediate_step import StreamEventData


@dataclass
class MiddlewareTrace:
    output: Any = None
    status: str = "ok"


@contextmanager
def trace_middleware(name: str,
                     *,
                     input_data: Any,
                     function_name: str,
                     guardrail: bool = False) -> Iterator[MiddlewareTrace]:
    """Trace one execution without changing the active function's ancestry.

    The end event is also emitted when execution raises or the caller closes a stream.
    """
    manager = Context.get().intermediate_step_manager
    start_type = IntermediateStepType.GUARDRAIL_START if guardrail else IntermediateStepType.SPAN_START
    end_type = IntermediateStepType.GUARDRAIL_END if guardrail else IntermediateStepType.SPAN_END
    start = IntermediateStepPayload(event_type=start_type,
                                    name=name,
                                    data=StreamEventData(input=input_data),
                                    metadata={"function": function_name})
    manager.push_intermediate_step(start)
    trace = MiddlewareTrace()
    try:
        yield trace
    except BaseException as error:
        trace.status = type(error).__name__
        raise
    finally:
        manager.push_intermediate_step(
            IntermediateStepPayload(event_type=end_type,
                                    UUID=start.UUID,
                                    name=name,
                                    span_event_timestamp=start.event_timestamp,
                                    data=StreamEventData(input=input_data, output=trace.output),
                                    metadata={"status": trace.status}))
