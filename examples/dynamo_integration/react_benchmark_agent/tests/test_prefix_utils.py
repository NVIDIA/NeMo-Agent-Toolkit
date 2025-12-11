# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for react_benchmark_agent prefix_utils module."""

import pytest
from unittest.mock import AsyncMock, MagicMock

from react_benchmark_agent.prefix_utils import (
    generate_benchmark_prefix,
    run_question_with_prefix,
    example_manual_eval_loop,
)
from nat.plugins.langchain.dynamo_prefix_headers import (
    get_current_prefix_id,
    clear_prefix_id,
)


class TestGenerateBenchmarkPrefix:
    """Tests for prefix ID generation."""
    
    def test_default_template(self):
        """Test prefix generation with default template."""
        prefix = generate_benchmark_prefix("banking_001")
        
        # Should contain question ID
        assert "banking_001" in prefix
        # Should start with react-benchmark
        assert prefix.startswith("react-benchmark-")
        # Should have UUID suffix (8 chars)
        parts = prefix.split("-")
        assert len(parts) >= 4  # react, benchmark, banking_001, uuid
    
    def test_custom_template(self):
        """Test prefix generation with custom template."""
        prefix = generate_benchmark_prefix(
            "q123",
            template="custom-{question_id}-{uuid}"
        )
        
        assert "custom-q123-" in prefix
        # Should have 8-char UUID
        uuid_part = prefix.split("-")[-1]
        assert len(uuid_part) == 8
    
    def test_unique_prefixes(self):
        """Test that multiple calls generate unique prefixes."""
        prefixes = [generate_benchmark_prefix("q1") for _ in range(10)]
        
        # All should be unique (due to different UUIDs)
        assert len(prefixes) == len(set(prefixes))
    
    def test_question_id_preserved(self):
        """Test that question ID is properly embedded in prefix."""
        question_id = "banking_transfer_scenario_42"
        prefix = generate_benchmark_prefix(question_id)
        
        assert question_id in prefix


class TestRunQuestionWithPrefix:
    """Tests for running questions with prefix management."""
    
    @pytest.fixture
    def mock_workflow(self):
        """Create a mock workflow function."""
        return AsyncMock(return_value={"result": "success"})
    
    @pytest.mark.asyncio
    async def test_sets_and_clears_prefix(self, mock_workflow):
        """Test that prefix is set and cleared properly."""
        # Ensure clean state
        clear_prefix_id()
        
        result = await run_question_with_prefix(
            question_id="test_q1",
            question_text="What is 2+2?",
            workflow_fn=mock_workflow
        )
        
        # Should return workflow result
        assert result == {"result": "success"}
        
        # Should call workflow once
        mock_workflow.assert_called_once_with("What is 2+2?")
        
        # Prefix should be cleared after execution
        assert get_current_prefix_id() is None
    
    @pytest.mark.asyncio
    async def test_prefix_set_during_execution(self, mock_workflow):
        """Test that prefix is available during workflow execution."""
        captured_prefix = None
        
        async def capture_prefix_workflow(*args, **kwargs):
            nonlocal captured_prefix
            captured_prefix = get_current_prefix_id()
            return {"result": "success"}
        
        await run_question_with_prefix(
            question_id="test_q2",
            question_text="Test question",
            workflow_fn=capture_prefix_workflow
        )
        
        # Prefix should have been set during execution
        assert captured_prefix is not None
        assert "test_q2" in captured_prefix
        
        # But should be cleared afterwards
        assert get_current_prefix_id() is None
    
    @pytest.mark.asyncio
    async def test_clears_prefix_on_exception(self, mock_workflow):
        """Test that prefix is cleared even if workflow raises exception."""
        mock_workflow.side_effect = RuntimeError("Workflow failed")
        
        with pytest.raises(RuntimeError, match="Workflow failed"):
            await run_question_with_prefix(
                question_id="test_q3",
                question_text="Test question",
                workflow_fn=mock_workflow
            )
        
        # Prefix should still be cleared
        assert get_current_prefix_id() is None
    
    @pytest.mark.asyncio
    async def test_workflow_kwargs_passed(self, mock_workflow):
        """Test that additional kwargs are passed to workflow."""
        await run_question_with_prefix(
            question_id="test_q4",
            question_text="Test question",
            workflow_fn=mock_workflow,
            extra_param="value",
            another_param=123
        )
        
        # Check that workflow was called with the kwargs
        mock_workflow.assert_called_once_with(
            "Test question",
            extra_param="value",
            another_param=123
        )


class TestExampleManualEvalLoop:
    """Tests for the example manual eval loop."""
    
    @pytest.mark.asyncio
    async def test_processes_all_questions(self):
        """Test that all questions are processed."""
        mock_workflow = MagicMock()
        mock_workflow.ainvoke = AsyncMock(return_value={"result": "ok"})
        
        questions = [
            ("q1", "Question 1"),
            ("q2", "Question 2"),
            ("q3", "Question 3"),
        ]
        
        results = await example_manual_eval_loop(mock_workflow, questions)
        
        # Should return results for all questions
        assert len(results) == 3
        assert all(r == {"result": "ok"} for r in results)
        
        # Should call workflow for each question
        assert mock_workflow.ainvoke.call_count == 3
    
    @pytest.mark.asyncio
    async def test_unique_prefix_per_question(self):
        """Test that each question gets a unique prefix."""
        captured_prefixes = []
        
        async def capture_prefix(*args, **kwargs):
            prefix = get_current_prefix_id()
            captured_prefixes.append(prefix)
            return {"result": "ok"}
        
        mock_workflow = MagicMock()
        mock_workflow.ainvoke = capture_prefix
        
        questions = [
            ("q1", "Question 1"),
            ("q2", "Question 2"),
            ("q3", "Question 3"),
        ]
        
        await example_manual_eval_loop(mock_workflow, questions)
        
        # All prefixes should be unique
        assert len(captured_prefixes) == 3
        assert len(set(captured_prefixes)) == 3
        
        # All should contain their question IDs
        assert any("q1" in p for p in captured_prefixes)
        assert any("q2" in p for p in captured_prefixes)
        assert any("q3" in p for p in captured_prefixes)
    
    @pytest.mark.asyncio
    async def test_prefix_cleaned_between_questions(self):
        """Test that prefix is properly cleaned between questions."""
        prefix_states = []
        
        async def check_prefix_state(*args, **kwargs):
            # Record prefix at start of each question
            prefix_states.append(get_current_prefix_id())
            return {"result": "ok"}
        
        mock_workflow = MagicMock()
        mock_workflow.ainvoke = check_prefix_state
        
        questions = [("q1", "Q1"), ("q2", "Q2")]
        
        await example_manual_eval_loop(mock_workflow, questions)
        
        # Each question should see a different prefix
        assert prefix_states[0] != prefix_states[1]
        
        # After loop, prefix should be cleared
        assert get_current_prefix_id() is None


class TestPrefixIntegration:
    """Integration tests for prefix management."""
    
    @pytest.mark.asyncio
    async def test_full_eval_scenario(self):
        """Test a complete evaluation scenario."""
        # Track all LLM calls with their prefixes
        llm_calls = []
        
        async def mock_llm_workflow(question, **kwargs):
            # Simulate multiple LLM calls per question
            for i in range(3):
                prefix = get_current_prefix_id()
                llm_calls.append({
                    "question": question,
                    "call_num": i,
                    "prefix": prefix
                })
            return {"answer": f"Processed: {question}"}
        
        # Run multiple questions
        questions = [
            ("banking_q1", "What is my balance?"),
            ("banking_q2", "Transfer $500"),
            ("banking_q3", "Check transaction history"),
        ]
        
        results = []
        for q_id, q_text in questions:
            result = await run_question_with_prefix(
                question_id=q_id,
                question_text=q_text,
                workflow_fn=mock_llm_workflow
            )
            results.append(result)
        
        # Verify results
        assert len(results) == 3
        
        # Verify LLM calls: 3 questions × 3 calls each = 9 total
        assert len(llm_calls) == 9
        
        # Group calls by prefix
        from collections import defaultdict
        prefix_groups = defaultdict(list)
        for call in llm_calls:
            prefix_groups[call["prefix"]].append(call)
        
        # Should have 3 unique prefixes (one per question)
        assert len(prefix_groups) == 3
        
        # Each prefix should have 3 calls (multi-turn conversation)
        for prefix, calls in prefix_groups.items():
            assert len(calls) == 3
            # All calls for same prefix should be from same question
            questions_in_group = {c["question"] for c in calls}
            assert len(questions_in_group) == 1
        
        # Final state should be clean
        assert get_current_prefix_id() is None

