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

# pylint: disable=import-outside-toplevel

from __future__ import annotations

import pytest

# Ensure registration of provider/clients from nat.test.llm
import nat.test.llm as _test_llm  # noqa: F401
from nat.builder.framework_enum import LLMFrameworkEnum
from nat.builder.workflow_builder import WorkflowBuilder
from nat.runtime.loader import load_workflow
from nat.test.llm import TestLLMConfig

RESP_SEQ = ["alpha", "beta", "gamma"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "seq,expected",
    [
        ([], ""),
        (["alpha"], "alpha"),
        (RESP_SEQ, "alpha"),
    ],
)
async def test_yaml_llm_chat_completion_single(tmp_path, seq, expected):
    """YAML e2e: first call returns first element (or empty if none)."""
    seq_yaml = ", ".join(seq)
    yaml_content = f"""
llms:
  main:
    _type: nat_test_llm
    response_seq: [{seq_yaml}]
    delay_ms: 0
workflow:
  _type: chat_completion
  llm_name: main
  system_prompt: "Say only the answer."
"""
    config_file = tmp_path / "chat_completion_single.yml"
    config_file.write_text(yaml_content)

    async with load_workflow(config_file) as workflow:
        async with workflow.run("What is 1+2?") as runner:
            result = await runner.result()
    from langchain_core.messages import AIMessage
    assert isinstance(result, AIMessage)
    assert result.content == expected


@pytest.mark.asyncio
@pytest.mark.parametrize("workflow_first", [True, False])
async def test_yaml_llm_chat_completion_cycle_and_ordering(tmp_path, workflow_first: bool):
    """YAML e2e: three calls cycle responses; validate both YAML key orderings."""
    yaml_workflow = """
workflow:
  _type: chat_completion
  llm_name: main
  system_prompt: "irrelevant"
llms:
  main:
    _type: nat_test_llm
    response_seq: [alpha, beta, gamma]
    delay_ms: 0
"""
    yaml_llms_first = """
llms:
  main:
    _type: nat_test_llm
    response_seq: [alpha, beta, gamma]
    delay_ms: 0
workflow:
  _type: chat_completion
  llm_name: main
  system_prompt: "irrelevant"
"""
    config_file = tmp_path / ("chat_completion_cycle_workflow_first.yml"
                              if workflow_first else "chat_completion_cycle_llms_first.yml")
    config_file.write_text(yaml_workflow if workflow_first else yaml_llms_first)

    async with load_workflow(config_file) as workflow:
        from langchain_core.messages import AIMessage
        async with workflow.run("a") as r1:
            out1 = await r1.result()
        async with workflow.run("b") as r2:
            out2 = await r2.result()
        async with workflow.run("c") as r3:
            out3 = await r3.result()

        assert isinstance(out1, AIMessage) and isinstance(out2, AIMessage) and isinstance(out3, AIMessage)
        assert [out1.content, out2.content, out3.content] == RESP_SEQ


@pytest.mark.asyncio
async def test_yaml_llm_chat_completion_with_delay(tmp_path):
    """YAML e2e: llm delay is respected; still returns first response."""
    yaml_content = """
llms:
  main:
    _type: nat_test_llm
    response_seq: [alpha, beta, gamma]
    delay_ms: 5
workflow:
  _type: chat_completion
  llm_name: main
  system_prompt: "irrelevant"
"""
    config_file = tmp_path / "chat_completion_delay.yml"
    config_file.write_text(yaml_content)

    async with load_workflow(config_file) as workflow:
        async with workflow.run("x") as runner:
            result = await runner.result()
    from langchain_core.messages import AIMessage
    assert isinstance(result, AIMessage)
    assert result.content == RESP_SEQ[0]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "seq_a,seq_b,exp_a,exp_b",
    [
        (RESP_SEQ, ["one", "two", "three"], "alpha", "one"),
        (["hello"], ["x"], "hello", "x"),
    ],
)
async def test_yaml_llm_chat_completion_two_configs(tmp_path, seq_a, seq_b, exp_a, exp_b):
    """YAML e2e: two different LLM configs yield different first outputs across loads."""
    a_yaml = ", ".join(seq_a)
    b_yaml = ", ".join(seq_b)
    yaml_a = f"""
llms:
  main:
    _type: nat_test_llm
    response_seq: [{a_yaml}]
    delay_ms: 0
workflow:
  _type: chat_completion
  llm_name: main
  system_prompt: "irrelevant"
"""
    yaml_b = f"""
llms:
  main:
    _type: nat_test_llm
    response_seq: [{b_yaml}]
    delay_ms: 0
workflow:
  _type: chat_completion
  llm_name: main
  system_prompt: "irrelevant"
"""
    file_a = tmp_path / "chat_completion_a.yml"
    file_b = tmp_path / "chat_completion_b.yml"
    file_a.write_text(yaml_a)
    file_b.write_text(yaml_b)

    from langchain_core.messages import AIMessage

    async with load_workflow(file_a) as wf_a:
        async with wf_a.run("p") as ra:
            out_a1 = await ra.result()
        assert isinstance(out_a1, AIMessage)
        assert out_a1.content == exp_a

    async with load_workflow(file_b) as wf_b:
        async with wf_b.run("p") as rb:
            out_b1 = await rb.result()
        assert isinstance(out_b1, AIMessage)
        assert out_b1.content == exp_b


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "seq,expected",
    [
        ([], ["", "", ""]),
        (["only"], ["only", "only", "only"]),
        (["a", "b"], ["a", "b", "a"]),
        (["x", "y", "z"], ["x", "y", "z"]),
    ],
)
async def test_yaml_llm_cycle_varied_lengths(tmp_path, seq, expected):
    """Different response_seq lengths cycle as expected, including empty."""
    seq_yaml = ", ".join(seq)
    yaml_content = f"""
llms:
  main:
    _type: nat_test_llm
    response_seq: [{seq_yaml}]
    delay_ms: 0
workflow:
  _type: chat_completion
  llm_name: main
  system_prompt: "irrelevant"
"""
    config_file = tmp_path / "chat_completion_varlen.yml"
    config_file.write_text(yaml_content)

    from langchain_core.messages import AIMessage

    async with load_workflow(config_file) as workflow:
        outs = []
        for prompt in ("p1", "p2", "p3"):
            async with workflow.run(prompt) as runner:
                res = await runner.result()
            assert isinstance(res, AIMessage)
            outs.append(res.content)

    assert outs == expected


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "seq",
    [
        ["hello, world!", "a:b", "c-d"],
        ["quote ' single", 'quote " double'],
    ],
)
async def test_yaml_llm_special_char_sequences(tmp_path, seq):
    """Special characters in YAML sequences are preserved and returned."""

    # Build YAML with proper quoting; use explicit list literal to avoid errors
    def _format_item(s: str) -> str:
        if '"' in s and "'" in s:
            # fallback to double quoting and escape inner quotes minimally
            return '"' + s.replace('"', '\\"') + '"'
        if '"' in s:
            return f"'{s}'"
        return f'"{s}"'

    seq_yaml = ", ".join(_format_item(s) for s in seq)
    yaml_content = f"""
llms:
  main:
    _type: nat_test_llm
    response_seq: [{seq_yaml}]
    delay_ms: 0
workflow:
  _type: chat_completion
  llm_name: main
  system_prompt: "irrelevant"
"""
    config_file = tmp_path / "chat_completion_special.yml"
    config_file.write_text(yaml_content)

    from langchain_core.messages import AIMessage

    async with load_workflow(config_file) as workflow:
        outs = []
        for prompt in ("p1", "p2", "p3"):
            async with workflow.run(prompt) as runner:
                res = await runner.result()
            assert isinstance(res, AIMessage)
            outs.append(res.content)

    # Only compare up to len(seq)
    assert outs[:len(seq)] == seq


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "seq,num_runs,expected",
    [
        (["a"], 5, ["a", "a", "a", "a", "a"]),
        (["a", "b"], 5, ["a", "b", "a", "b", "a"]),
    ],
)
async def test_yaml_llm_cycle_persistence_across_runs(tmp_path, seq, num_runs, expected):
    """Cycle persists across many runs within the same loaded workflow."""
    seq_yaml = ", ".join(seq)
    yaml_content = f"""
llms:
  main:
    _type: nat_test_llm
    response_seq: [{seq_yaml}]
    delay_ms: 0
workflow:
  _type: chat_completion
  llm_name: main
  system_prompt: "irrelevant"
"""
    config_file = tmp_path / "chat_completion_many.yml"
    config_file.write_text(yaml_content)

    from langchain_core.messages import AIMessage

    async with load_workflow(config_file) as workflow:
        outs = []
        for i in range(num_runs):
            async with workflow.run(f"p{i}") as runner:
                res = await runner.result()
            assert isinstance(res, AIMessage)
            outs.append(res.content)

    assert outs == expected


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "wrapper, seq",
    [
        (LLMFrameworkEnum.LANGCHAIN.value, ["a", "b", "c"]),
        (LLMFrameworkEnum.LLAMA_INDEX.value, ["x", "y", "z"]),
        (LLMFrameworkEnum.CREWAI.value, ["p", "q", "r"]),
        (LLMFrameworkEnum.SEMANTIC_KERNEL.value, ["s1", "s2", "s3"]),
        (LLMFrameworkEnum.AGNO.value, ["m", "n", "o"]),
    ],
)
async def test_builder_framework_cycle(wrapper: str, seq: list[str]):
    """Build workflows programmatically and validate per-framework cycle order."""

    if wrapper == LLMFrameworkEnum.SEMANTIC_KERNEL.value:
        pytest.importorskip("semantic_kernel")

    async with WorkflowBuilder() as builder:
        cfg = TestLLMConfig(response_seq=list(seq), delay_ms=0)
        await builder.add_llm("main", cfg)
        client = await builder.get_llm("main", wrapper_type=wrapper)

        outs: list[str] = []

        if wrapper == LLMFrameworkEnum.LANGCHAIN.value:
            from langchain_core.messages import AIMessage

            for i in range(len(seq)):
                res = await client.ainvoke([
                    {
                        "role": "user", "content": f"p{i}"
                    },
                ])
                assert isinstance(res, AIMessage)
                outs.append(str(res.content))

        elif wrapper == LLMFrameworkEnum.LLAMA_INDEX.value:
            for _ in range(len(seq)):
                r = await client.achat([])
                assert hasattr(r, "text"), f"Response {r} missing 'text' attribute"
                outs.append(r.text)
        elif wrapper == LLMFrameworkEnum.CREWAI.value:
            for i in range(len(seq)):
                r = client.call([
                    {
                        "role": "user", "content": f"p{i}"
                    },
                ])
                assert isinstance(r, str)
                outs.append(r)

        elif wrapper == LLMFrameworkEnum.SEMANTIC_KERNEL.value:
            from semantic_kernel.contents.chat_message_content import ChatMessageContent

            for _ in range(len(seq)):
                lst = await client.get_chat_message_contents(chat_history=None)
                assert isinstance(lst, list) and len(lst) == 1
                assert isinstance(lst[0], ChatMessageContent)
                outs.append(str(lst[0].content))

        elif wrapper == LLMFrameworkEnum.AGNO.value:
            for i in range(len(seq)):
                r = await client.ainvoke([
                    {
                        "role": "user", "content": f"p{i}"
                    },
                ])
                # Agno client returns str in our test client
                assert isinstance(r, str)
                outs.append(r)

        else:
            pytest.skip(f"Unsupported wrapper: {wrapper}")

    assert outs == seq
