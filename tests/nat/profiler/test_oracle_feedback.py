# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from nat.profiler.parameter_optimization.oracle_feedback import build_oracle_feedback


class TestBuildOracleFeedback:
    """Tests for build_oracle_feedback function."""

    def test_empty_reasoning_returns_none(self):
        """Returns None when no reasoning provided."""
        result = build_oracle_feedback([], max_chars=4000)
        assert result is None

    def test_single_reasoning(self):
        """Formats single reasoning item correctly."""
        result = build_oracle_feedback(["Failed to answer question"], max_chars=4000)
        assert result == "1. Failed to answer question\n"

    def test_multiple_reasoning(self):
        """Formats multiple reasoning items with numbers."""
        reasons = ["First failure", "Second failure", "Third failure"]
        result = build_oracle_feedback(reasons, max_chars=4000)
        assert result == "1. First failure\n2. Second failure\n3. Third failure\n"

    def test_truncation_at_char_limit(self):
        """Truncates reasoning to fit within max_chars."""
        reasons = ["A" * 100, "B" * 100, "C" * 100]
        result = build_oracle_feedback(reasons, max_chars=120)
        # Should include first item and partial second
        assert result is not None
        assert len(result) <= 120
        assert "1. " in result
        assert "..." in result  # Truncation indicator

    def test_skips_entry_if_no_meaningful_space(self):
        """Skips entries when remaining space is too small."""
        reasons = ["A" * 50]
        result = build_oracle_feedback(reasons, max_chars=10)
        # Not enough space for even "1. " + content
        assert result is None or len(result) <= 10

    def test_preserves_evaluator_labels(self):
        """Preserves evaluator labels in reasoning."""
        reasons = ["[Accuracy] Score too low", "[Relevance] Off topic"]
        result = build_oracle_feedback(reasons, max_chars=4000)
        assert "[Accuracy]" in result
        assert "[Relevance]" in result
