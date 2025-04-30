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

import asyncio
import typing
from collections.abc import AsyncGenerator

from aiq.data_models.api_server import AIQResponseIntermediateStep
from aiq.data_models.api_server import AIQResponsePayloadOutput
from aiq.data_models.api_server import AIQResponseSerializable
from aiq.data_models.step_adaptor import StepAdaptorConfig
from aiq.front_ends.fastapi.intermediate_steps_subscriber import pull_intermediate
from aiq.front_ends.fastapi.step_adaptor import StepAdaptor
from aiq.runtime.session import AIQSessionManager
from aiq.utils.producer_consumer_queue import AsyncIOProducerConsumerQueue


async def generate_streaming_response_as_str(payload: typing.Any,
                                             *,
                                             session_manager: AIQSessionManager,
                                             streaming: bool,
                                             step_adaptor: StepAdaptor = StepAdaptor(StepAdaptorConfig()),
                                             result_type: type | None = None,
                                             output_type: type | None = None) -> AsyncGenerator[str]:

    async for item in generate_streaming_response(payload,
                                                  session_manager=session_manager,
                                                  streaming=streaming,
                                                  step_adaptor=step_adaptor,
                                                  result_type=result_type,
                                                  output_type=output_type):

        if (isinstance(item, AIQResponseSerializable)):
            yield item.get_stream_data()
        else:
            raise ValueError("Unexpected item type in stream. Expected AIQChatResponseSerializable, got: " +
                             str(type(item)))


async def generate_streaming_response(payload: typing.Any,
                                      *,
                                      session_manager: AIQSessionManager,
                                      streaming: bool,
                                      step_adaptor: StepAdaptor = StepAdaptor(StepAdaptorConfig()),
                                      result_type: type | None = None,
                                      output_type: type | None = None) -> AsyncGenerator[AIQResponseSerializable]:

    async with session_manager.run(payload) as runner:

        q: AsyncIOProducerConsumerQueue[AIQResponseSerializable] = AsyncIOProducerConsumerQueue()

        # Start the intermediate stream
        intermediate_complete = await pull_intermediate(q, step_adaptor)

        async def pull_result():
            if session_manager.workflow.has_streaming_output and streaming:
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

                if (isinstance(item, AIQResponseSerializable)):
                    yield item
                else:
                    yield AIQResponsePayloadOutput(payload=item)
        except Exception as e:
            # Handle exceptions here
            raise e
        finally:
            await q.close()


async def generate_single_response(
    payload: typing.Any,
    session_manager: AIQSessionManager,
    result_type: type | None = None,
) -> typing.Any:
    if (not session_manager.workflow.has_single_output):
        raise ValueError("Cannot get a single output value for streaming workflows")

    async with session_manager.run(payload) as runner:
        return await runner.result(to_type=result_type)


async def generate_streaming_response_full(payload: typing.Any,
                                           *,
                                           session_manager: AIQSessionManager,
                                           streaming: bool,
                                           result_type: type | None = None,
                                           output_type: type | None = None,
                                           filter_steps: str | None = None) -> AsyncGenerator[AIQResponseSerializable]:
    """
    Similar to generate_streaming_response but provides raw AIQResponseIntermediateStep objects
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

    async with session_manager.run(payload) as runner:
        q: AsyncIOProducerConsumerQueue[AIQResponseSerializable] = AsyncIOProducerConsumerQueue()

        # Start the intermediate stream without step adaptor
        intermediate_complete = await pull_intermediate(q, None)

        async def pull_result():
            if session_manager.workflow.has_streaming_output and streaming:
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
                if (isinstance(item, AIQResponseIntermediateStep)):
                    # Filter intermediate steps if filter_steps is provided
                    if allowed_types is None or item.type in allowed_types:
                        yield item
                else:
                    yield AIQResponsePayloadOutput(payload=item)
        except Exception as e:
            # Handle exceptions here
            raise e
        finally:
            await q.close()


async def generate_streaming_response_full_as_str(payload: typing.Any,
                                                  *,
                                                  session_manager: AIQSessionManager,
                                                  streaming: bool,
                                                  result_type: type | None = None,
                                                  output_type: type | None = None,
                                                  filter_steps: str | None = None) -> AsyncGenerator[str]:
    """
    Similar to generate_streaming_response but converts the response to a string format.
    """
    async for item in generate_streaming_response_full(payload,
                                                       session_manager=session_manager,
                                                       streaming=streaming,
                                                       result_type=result_type,
                                                       output_type=output_type,
                                                       filter_steps=filter_steps):
        if (isinstance(item, AIQResponseIntermediateStep) or isinstance(item, AIQResponsePayloadOutput)):
            yield item.get_stream_data()
        else:
            raise ValueError("Unexpected item type in stream. Expected AIQChatResponseSerializable, got: " +
                             str(type(item)))
