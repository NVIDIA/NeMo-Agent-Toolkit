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
"""A LangChain workflow reports its LLM calls as intermediate steps without the optional profiler package."""

import json
import subprocess
import sys
import textwrap

import pytest

# Runs in a fresh interpreter: the builder decides how to hook frameworks when it's imported, so the profiler has to
# be unavailable before anything from NAT is imported.
_SCRIPT = textwrap.dedent('''
    import sys
    if sys.argv[1] == "without_profiler":
        sys.modules["nat.plugins.profiler"] = None  # as if nvidia-nat-profiler weren't installed
    EXTRA_HANDLER = sys.argv[2] == "extra_handler"
    MODEL_HANDLER = sys.argv[2] == "model_handler"

    import asyncio
    import json

    from langchain_core.language_models.fake_chat_models import FakeListChatModel

    from nat.builder.builder import Builder
    from nat.builder.context import Context
    from nat.builder.framework_enum import LLMFrameworkEnum
    from nat.builder.function_info import FunctionInfo
    from nat.builder.llm import LLMProviderInfo
    from nat.builder.workflow_builder import WorkflowBuilder
    from nat.cli.register_workflow import register_function, register_llm_client, register_llm_provider
    from nat.data_models.config import Config
    from nat.data_models.function import FunctionBaseConfig
    from nat.data_models.llm import LLMBaseConfig
    from nat.runtime.session import SessionManager

    class FakeChatConfig(LLMBaseConfig, name="instrumentation_test_fake_chat"):
        pass

    @register_llm_provider(config_type=FakeChatConfig)
    async def _provider(config, builder: Builder):
        yield LLMProviderInfo(config=config, description="fake chat model")

    @register_llm_client(config_type=FakeChatConfig, wrapper_type=LLMFrameworkEnum.LANGCHAIN)
    async def _client(config, builder: Builder):
        if MODEL_HANDLER:
            # a handler attached to the model itself, next to the one the workflow attaches
            from nat.plugins.langchain.callback_handler import LangchainProfilerHandler
            yield FakeListChatModel(responses=["Paris."], callbacks=[LangchainProfilerHandler()])
        else:
            yield FakeListChatModel(responses=["Paris."])

    class AskConfig(FunctionBaseConfig, name="instrumentation_test_ask"):
        llm_name: str = "chat"

    @register_function(config_type=AskConfig, framework_wrappers=[LLMFrameworkEnum.LANGCHAIN])
    async def _ask(config, builder: Builder):
        llm = await builder.get_llm(config.llm_name, wrapper_type=LLMFrameworkEnum.LANGCHAIN)

        async def _inner(question: str) -> str:
            if EXTRA_HANDLER:
                # as sequential_executor and the guardrails handler do: an explicit handler next to the workflow's
                from nat.plugins.langchain.callback_handler import LangchainProfilerHandler
                return (await llm.ainvoke(question, config={"callbacks": [LangchainProfilerHandler()]})).content
            return (await llm.ainvoke(question)).content

        yield FunctionInfo.from_fn(_inner, description="asks the LLM")

    async def main():
        config = Config.model_validate({"llms": {"chat": {"_type": "instrumentation_test_fake_chat"}},
                                        "workflow": {"_type": "instrumentation_test_ask"}})
        steps = []
        async with WorkflowBuilder.from_config(config=config) as builder:
            session_manager = await SessionManager.create(config=config, shared_builder=builder)
            async with session_manager.session() as session:
                async with session.run("Capital of France?") as runner:
                    Context.get().intermediate_step_manager.subscribe(lambda step: steps.append(step.event_type.value))
                    answer = await runner.result(to_type=str)
            await session_manager.shutdown()
        print(json.dumps({"answer": answer, "steps": steps}))

    asyncio.run(main())
''')


def _run(mode: str, handlers: str = "workflow_handler") -> dict:
    completed = subprocess.run([sys.executable, "-c", _SCRIPT, mode, handlers],
                               capture_output=True,
                               text=True,
                               timeout=120,
                               check=False)
    assert completed.returncode == 0, completed.stderr[-3000:]
    return json.loads(completed.stdout.strip().splitlines()[-1])


def test_llm_calls_are_reported_without_the_profiler():
    result = _run("without_profiler")
    assert result["answer"] == "Paris."
    assert result["steps"].count("LLM_START") == 1
    assert result["steps"].count("LLM_END") == 1


@pytest.mark.parametrize("handlers", ["extra_handler", "model_handler"], ids=["call_callbacks", "model_callbacks"])
def test_an_explicit_handler_does_not_duplicate_llm_steps_without_the_profiler(handlers):
    """The LangChain handler defers to the one the workflow attached, so each call is reported once."""
    result = _run("without_profiler", handlers)
    assert result["answer"] == "Paris."
    assert result["steps"].count("LLM_START") == 1
    assert result["steps"].count("LLM_END") == 1


def test_llm_calls_are_reported_once_with_the_profiler_installed():
    pytest.importorskip("nat.plugins.profiler")
    result = _run("as_installed")
    assert result["answer"] == "Paris."
    assert result["steps"].count("LLM_START") == 1
    assert result["steps"].count("LLM_END") == 1
