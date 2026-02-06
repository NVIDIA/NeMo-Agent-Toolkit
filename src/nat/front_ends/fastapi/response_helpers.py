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

import asyncio
import typing
from collections.abc import AsyncGenerator

from nat.data_models.api_server import ResponseIntermediateStep
from nat.data_models.api_server import ResponsePayloadOutput
from nat.data_models.api_server import ResponseSerializable
from nat.data_models.step_adaptor import StepAdaptorConfig
from nat.front_ends.fastapi.intermediate_steps_subscriber import pull_intermediate
from nat.front_ends.fastapi.step_adaptor import StepAdaptor
from nat.runtime.session import Session
from nat.utils.producer_consumer_queue import AsyncIOProducerConsumerQueue


async def generate_streaming_response_as_str(payload: typing.Any,
                                             *,
                                             session: Session,
                                             streaming: bool,
                                             step_adaptor: StepAdaptor = StepAdaptor(StepAdaptorConfig()),
                                             result_type: type | None = None,
                                             output_type: type | None = None) -> AsyncGenerator[str]:

    async for item in generate_streaming_response(payload,
                                                  session=session,
                                                  streaming=streaming,
                                                  step_adaptor=step_adaptor,
                                                  result_type=result_type,
                                                  output_type=output_type):

        if (isinstance(item, ResponseSerializable)):
            yield item.get_stream_data()
        else:
            raise ValueError("Unexpected item type in stream. Expected ChatResponseSerializable, got: " +
                             str(type(item)))


async def generate_streaming_response(payload: typing.Any,
                                      *,
                                      session: Session,
                                      streaming: bool,
                                      step_adaptor: StepAdaptor = StepAdaptor(StepAdaptorConfig()),
                                      result_type: type | None = None,
                                      output_type: type | None = None) -> AsyncGenerator[ResponseSerializable]:

    async with session.run(payload) as runner:

        q: AsyncIOProducerConsumerQueue[ResponseSerializable] = AsyncIOProducerConsumerQueue()

        # Start the intermediate stream
        intermediate_complete = await pull_intermediate(q, step_adaptor)

        async def pull_result():
            if session.workflow.has_streaming_output and streaming:
                async for chunk in runner.result_stream(to_type=output_type):
                    await q.put(chunk)
            else:
                result = await runner.result(to_type=result_type)
                await q.put(runner.convert(result, output_type))

            # Wait until the intermediate subscription is done before closing q
            # But we have no direct "intermediate_done" reference here
            # because it's encapsulated in pull_intermediate. So we can do:
            #    await some_event.wait()
            # If needed. Alternatively, you can skip that if the intermediate
            # subscriber won't block the main flow.
            #
            # For example, if you *need* to guarantee the subscriber is done before
            # closing the queue, you can structure the code to store or return
            # the 'intermediate_done' event from pull_intermediate.
            #

            await intermediate_complete.wait()

            await q.close()

        try:
            # Start the result stream
            asyncio.create_task(pull_result())

            async for item in q:

                if (isinstance(item, ResponseSerializable)):
                    yield item
                else:
                    yield ResponsePayloadOutput(payload=item)
        except Exception:
            # Handle exceptions here
            raise
        finally:
            await q.close()


async def generate_single_response(
    payload: typing.Any,
    session: Session,
    result_type: type | None = None,
) -> typing.Any:

    if not session.workflow.has_single_output:
        raise ValueError("Cannot get a single output value for streaming workflows")

    async with session.run(payload) as runner:
        return await runner.result(to_type=result_type)


async def generate_streaming_response_full(payload: typing.Any,
                                           *,
                                           session: Session,
                                           streaming: bool,
                                           result_type: type | None = None,
                                           output_type: type | None = None,
                                           filter_steps: str | None = None) -> AsyncGenerator[ResponseSerializable]:
    """
    Similar to generate_streaming_response but provides raw ResponseIntermediateStep objects
    without any step adaptor translations.
    """
    # Parse filter_steps into a set of allowed types if provided
    # Special case: if filter_steps is "none", suppress all steps
    allowed_types = None
    if filter_steps:
        if filter_steps.lower() == "none":
            allowed_types = set()  # Empty set means no steps allowed
        else:
            allowed_types = set(filter_steps.split(','))

    async with session.run(payload) as runner:
        q: AsyncIOProducerConsumerQueue[ResponseSerializable] = AsyncIOProducerConsumerQueue()

        # Start the intermediate stream without step adaptor
        intermediate_complete = await pull_intermediate(q, None)

        async def pull_result():
            if session.workflow.has_streaming_output and streaming:
                async for chunk in runner.result_stream(to_type=output_type):
                    await q.put(chunk)
            else:
                result = await runner.result(to_type=result_type)
                await q.put(runner.convert(result, output_type))

            await intermediate_complete.wait()
            await q.close()

        try:
            # Start the result stream
            asyncio.create_task(pull_result())

            async for item in q:
                if (isinstance(item, ResponseIntermediateStep)):
                    # Filter intermediate steps if filter_steps is provided
                    if allowed_types is None or item.type in allowed_types:
                        yield item
                else:
                    yield ResponsePayloadOutput(payload=item)
        except Exception:
            # Handle exceptions here
            raise
        finally:
            await q.close()


async def generate_streaming_response_full_as_str(payload: typing.Any,
                                                  *,
                                                  session: Session,
                                                  streaming: bool,
                                                  result_type: type | None = None,
                                                  output_type: type | None = None,
                                                  filter_steps: str | None = None) -> AsyncGenerator[str]:
    """
    Similar to generate_streaming_response but converts the response to a string format.
    """
    async for item in generate_streaming_response_full(payload,
                                                       session=session,
                                                       streaming=streaming,
                                                       result_type=result_type,
                                                       output_type=output_type,
                                                       filter_steps=filter_steps):
        if (isinstance(item, ResponseIntermediateStep) or isinstance(item, ResponsePayloadOutput)):
            yield item.get_stream_data()
        else:
            raise ValueError("Unexpected item type in stream. Expected ChatResponseSerializable, got: " +
                             str(type(item)))


async def generate_responses_api_streaming(
    payload: typing.Any,
    *,
    session: Session,
    model: str,
    step_adaptor: StepAdaptor = StepAdaptor(StepAdaptorConfig()),
) -> AsyncGenerator[str, None]:
    """
    Generate streaming response in OpenAI Responses API format.
    Converts internal streaming chunks to Responses API event stream format.

    The Responses API uses Server-Sent Events with specific event types:
    - response.created: Initial response creation
    - response.output_item.added: New output item started
    - response.content_part.added: New content part started
    - response.output_text.delta: Text content delta
    - response.output_text.done: Text content completed
    - response.content_part.done: Content part completed
    - response.output_item.done: Output item completed
    - response.done: Full response completed
    - response.failed: Emitted when an error occurs during processing

    WARNING: This streaming format is provided for pass-through compatibility with managed
    services that support stateful backends. NAT agents do not inherently support stateful
    backends. The payload is processed through the NAT workflow and output is formatted
    to match the Responses API streaming specification.
    """
    import json
    import logging
    import uuid as uuid_module

    from nat.data_models.api_server import ChatResponse
    from nat.data_models.api_server import ChatResponseChunk

    _logger = logging.getLogger(__name__)

    response_id = f"resp_{uuid_module.uuid4().hex}{uuid_module.uuid4().hex}"[:50]
    message_id = f"msg_{uuid_module.uuid4().hex[:24]}"
    output_index = 0
    content_index = 0
    accumulated_text = ""
    error_occurred = False

    def _sse_event(event_type: str, data: dict) -> str:
        """Format a Server-Sent Event."""
        return f"event: {event_type}\ndata: {json.dumps(data)}\n\n"

    def _error_event(error_message: str, error_code: str = "server_error") -> str:
        """Generate a response.failed SSE event."""
        return _sse_event("response.failed", {
            "type": "response.failed",
            "response": {
                "id": response_id,
                "object": "response",
                "status": "failed",
                "model": model,
                "error": {
                    "type": error_code,
                    "message": error_message,
                },
                "output": [],
            }
        })

    try:
        # Event: response.created
        yield _sse_event("response.created", {
            "type": "response.created",
            "response": {
                "id": response_id,
                "object": "response",
                "status": "in_progress",
                "model": model,
                "output": [],
            }
        })

        # Event: response.output_item.added
        yield _sse_event("response.output_item.added", {
            "type": "response.output_item.added",
            "output_index": output_index,
            "item": {
                "type": "message",
                "id": message_id,
                "status": "in_progress",
                "role": "assistant",
                "content": [],
            }
        })

        # Event: response.content_part.added
        yield _sse_event("response.content_part.added", {
            "type": "response.content_part.added",
            "output_index": output_index,
            "content_index": content_index,
            "part": {
                "type": "output_text",
                "text": "",
            }
        })

        # Process the workflow and stream text deltas
        try:
            async with session.run(payload) as runner:
                if session.workflow.has_streaming_output:
                    async for chunk in runner.result_stream(to_type=ChatResponseChunk):
                        # Extract text from ChatResponseChunk
                        if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                            delta_text = chunk.choices[0].delta.content
                            accumulated_text += delta_text

                            # Event: response.output_text.delta
                            yield _sse_event("response.output_text.delta", {
                                "type": "response.output_text.delta",
                                "output_index": output_index,
                                "content_index": content_index,
                                "delta": delta_text,
                            })
                else:
                    # Non-streaming workflow - get full result and emit as single delta
                    result = await runner.result(to_type=ChatResponse)
                    if result.choices and result.choices[0].message:
                        accumulated_text = result.choices[0].message.content or ""
                        if accumulated_text:
                            yield _sse_event("response.output_text.delta", {
                                "type": "response.output_text.delta",
                                "output_index": output_index,
                                "content_index": content_index,
                                "delta": accumulated_text,
                            })
        except Exception as workflow_error:
            error_occurred = True
            _logger.exception("Error during Responses API streaming workflow execution")
            yield _error_event(str(workflow_error), "workflow_error")
            return  # Stop streaming after error

        # Only emit completion events if no error occurred
        if not error_occurred:
            # Event: response.output_text.done
            yield _sse_event("response.output_text.done", {
                "type": "response.output_text.done",
                "output_index": output_index,
                "content_index": content_index,
                "text": accumulated_text,
            })

            # Event: response.content_part.done
            yield _sse_event("response.content_part.done", {
                "type": "response.content_part.done",
                "output_index": output_index,
                "content_index": content_index,
                "part": {
                    "type": "output_text",
                    "text": accumulated_text,
                }
            })

            # Event: response.output_item.done
            yield _sse_event("response.output_item.done", {
                "type": "response.output_item.done",
                "output_index": output_index,
                "item": {
                    "type": "message",
                    "id": message_id,
                    "status": "completed",
                    "role": "assistant",
                    "content": [{
                        "type": "output_text",
                        "text": accumulated_text,
                    }],
                }
            })

            # Event: response.done
            yield _sse_event("response.done", {
                "type": "response.done",
                "response": {
                    "id": response_id,
                    "object": "response",
                    "status": "completed",
                    "model": model,
                    "output": [{
                        "type": "message",
                        "id": message_id,
                        "status": "completed",
                        "role": "assistant",
                        "content": [{
                            "type": "output_text",
                            "text": accumulated_text,
                        }],
                    }],
                }
            })

    except Exception as setup_error:
        # Catch any errors during initial event emission (before workflow starts)
        _logger.exception("Error during Responses API streaming setup")
        yield _error_event(str(setup_error), "server_error")
