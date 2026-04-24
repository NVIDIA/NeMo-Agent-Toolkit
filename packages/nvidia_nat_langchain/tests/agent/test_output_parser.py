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

import pytest
from langchain_core.agents import AgentAction
from langchain_core.agents import AgentFinish


@pytest.fixture(name='parser', scope='module')
def fixture_parser():
    from nat.plugins.langchain.agent.react_agent.output_parser import ReActOutputParser
    return ReActOutputParser()


class TestTryParseJsonAction:
    """Tests for _try_parse_json_action — the JSON-format tool-call detector."""

    def test_dict_action_input_returns_agent_action(self, parser):
        text = '{"action": "my_tool", "action_input": {"query": "hello"}}'
        result = parser._try_parse_json_action(text)
        assert isinstance(result, AgentAction)
        assert result.tool == "my_tool"
        assert result.tool_input == {"query": "hello"}

    def test_string_action_input_returns_agent_action(self, parser):
        text = '{"action": "my_tool", "action_input": "simple string"}'
        result = parser._try_parse_json_action(text)
        assert isinstance(result, AgentAction)
        assert result.tool == "my_tool"
        assert result.tool_input == "simple string"

    def test_list_action_input_serialized_to_json_string(self, parser):
        # AgentAction.tool_input only accepts str | dict; lists are serialized to a JSON string
        text = '{"action": "bulk_op", "action_input": ["a", "b", "c"]}'
        result = parser._try_parse_json_action(text)
        assert isinstance(result, AgentAction)
        assert result.tool == "bulk_op"
        import json as _json
        assert result.tool_input == _json.dumps(["a", "b", "c"])

    def test_final_answer_action_returns_agent_finish(self, parser):
        text = '{"action": "Final Answer", "action_input": "The sky is blue"}'
        result = parser._try_parse_json_action(text)
        assert isinstance(result, AgentFinish)
        assert result.return_values["output"] == "The sky is blue"

    def test_final_answer_case_insensitive(self, parser):
        text = '{"action": "final answer", "action_input": "Some answer"}'
        result = parser._try_parse_json_action(text)
        assert isinstance(result, AgentFinish)
        assert result.return_values["output"] == "Some answer"

    def test_markdown_json_fence_unwrapped(self, parser):
        text = '```json\n{"action": "my_tool", "action_input": {"q": "test"}}\n```'
        result = parser._try_parse_json_action(text)
        assert isinstance(result, AgentAction)
        assert result.tool == "my_tool"
        assert result.tool_input == {"q": "test"}

    def test_plain_code_fence_unwrapped(self, parser):
        text = '```\n{"action": "my_tool", "action_input": "value"}\n```'
        result = parser._try_parse_json_action(text)
        assert isinstance(result, AgentAction)
        assert result.tool == "my_tool"
        assert result.tool_input == "value"

    def test_leading_trailing_whitespace_stripped(self, parser):
        text = '  \n  {"action": "my_tool", "action_input": "value"}  \n  '
        result = parser._try_parse_json_action(text)
        assert isinstance(result, AgentAction)
        assert result.tool == "my_tool"

    def test_action_name_whitespace_stripped(self, parser):
        text = '{"action": "  my_tool  ", "action_input": "value"}'
        result = parser._try_parse_json_action(text)
        assert isinstance(result, AgentAction)
        assert result.tool == "my_tool"

    def test_log_preserves_original_text(self, parser):
        text = '{"action": "my_tool", "action_input": "value"}'
        result = parser._try_parse_json_action(text)
        assert isinstance(result, AgentAction)
        assert result.log == text

    def test_missing_action_key_returns_none(self, parser):
        text = '{"tool": "my_tool", "action_input": "value"}'
        assert parser._try_parse_json_action(text) is None

    def test_missing_action_input_key_returns_none(self, parser):
        text = '{"action": "my_tool", "input": "value"}'
        assert parser._try_parse_json_action(text) is None

    def test_invalid_json_returns_none(self, parser):
        assert parser._try_parse_json_action("not json at all") is None

    def test_json_array_returns_none(self, parser):
        text = '[{"action": "my_tool", "action_input": "value"}]'
        assert parser._try_parse_json_action(text) is None

    def test_empty_string_returns_none(self, parser):
        assert parser._try_parse_json_action("") is None

    def test_partial_json_returns_none(self, parser):
        assert parser._try_parse_json_action('{"action": "tool"') is None


# =============================================================================
# Integration tests: parse() with JSON-format tool calls
# =============================================================================


class TestParseJsonFormat:
    """Tests for parse() / aparse() when the LLM emits JSON-format tool calls."""

    def test_nemotron_style_multi_line_tool_call(self, parser):
        """Exact format emitted by Nemotron that triggered the original bug."""
        text = ('{\n'
                '  "action": "mcp_outlook__outlook_list_messages",\n'
                '  "action_input": {\n'
                '    "query": "from:ddurst@nvidia.com"\n'
                '  }\n'
                '}')
        result = parser.parse(text)
        assert isinstance(result, AgentAction)
        assert result.tool == "mcp_outlook__outlook_list_messages"
        assert result.tool_input == {"query": "from:ddurst@nvidia.com"}

    def test_compact_json_tool_call(self, parser):
        text = '{"action": "search_tool", "action_input": {"q": "nvidia"}}'
        result = parser.parse(text)
        assert isinstance(result, AgentAction)
        assert result.tool == "search_tool"
        assert result.tool_input == {"q": "nvidia"}

    def test_json_tool_call_is_not_treated_as_direct_answer(self, parser):
        """JSON tool calls must not fall through to the 'missing action' direct-answer path."""
        text = '{"action": "some_tool", "action_input": {"key": "val"}}'
        result = parser.parse(text)
        assert isinstance(result, AgentAction)

    def test_json_final_answer(self, parser):
        text = '{"action": "Final Answer", "action_input": "The answer is 42"}'
        result = parser.parse(text)
        assert isinstance(result, AgentFinish)
        assert result.return_values["output"] == "The answer is 42"

    def test_json_fenced_tool_call(self, parser):
        text = '```json\n{"action": "my_tool", "action_input": {"q": "test"}}\n```'
        result = parser.parse(text)
        assert isinstance(result, AgentAction)
        assert result.tool == "my_tool"

    async def test_aparse_json_tool_call(self, parser):
        """aparse delegates to parse; verify the async path handles JSON too."""
        text = '{"action": "async_tool", "action_input": {"x": 1}}'
        result = await parser.aparse(text)
        assert isinstance(result, AgentAction)
        assert result.tool == "async_tool"
        assert result.tool_input == {"x": 1}

    async def test_aparse_json_final_answer(self, parser):
        text = '{"action": "Final Answer", "action_input": "done"}'
        result = await parser.aparse(text)
        assert isinstance(result, AgentFinish)
        assert result.return_values["output"] == "done"

    def test_json_takes_priority_over_text_regex(self, parser):
        """When input is valid JSON with both keys, the JSON path wins."""
        text = '{"action": "json_tool", "action_input": "json_value"}'
        result = parser.parse(text)
        assert isinstance(result, AgentAction)
        assert result.tool == "json_tool"
        assert result.tool_input == "json_value"

    def test_non_json_text_raises_missing_action(self, parser):
        """Plain text without Action: still raises the expected error."""
        from nat.plugins.langchain.agent.react_agent.output_parser import MISSING_ACTION_AFTER_THOUGHT_ERROR_MESSAGE
        from nat.plugins.langchain.agent.react_agent.output_parser import ReActOutputParserException

        with pytest.raises(ReActOutputParserException) as exc_info:
            parser.parse("I don't know what to do")
        assert exc_info.value.missing_action is True
        assert exc_info.value.observation == MISSING_ACTION_AFTER_THOUGHT_ERROR_MESSAGE

    def test_partial_json_falls_through_to_text_parser(self, parser):
        """Malformed JSON that also lacks text Action: raises missing_action."""
        from nat.plugins.langchain.agent.react_agent.output_parser import ReActOutputParserException

        with pytest.raises(ReActOutputParserException) as exc_info:
            parser.parse('{"action": "tool"')
        assert exc_info.value.missing_action is True

    def test_json_without_action_input_falls_through(self, parser):
        """JSON missing action_input falls through to text parsing and raises."""
        from nat.plugins.langchain.agent.react_agent.output_parser import ReActOutputParserException

        with pytest.raises(ReActOutputParserException):
            parser.parse('{"action": "tool", "other_key": "value"}')

    def test_text_format_still_works_after_json_check(self, parser):
        """Existing text-format tool calls must not be broken by the JSON check."""
        text = 'Thought: I should search\nAction: search_tool\nAction Input: hello world\nObservation:'
        result = parser.parse(text)
        assert isinstance(result, AgentAction)
        assert result.tool == "search_tool"
        assert result.tool_input == "hello world"

    def test_text_final_answer_still_works(self, parser):
        """Existing text-format Final Answer must not be broken by the JSON check."""
        result = parser.parse("Final Answer: lorem ipsum")
        assert isinstance(result, AgentFinish)
        assert result.return_values["output"] == "lorem ipsum"
