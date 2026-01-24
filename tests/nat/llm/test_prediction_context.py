# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


from nat.llm.prediction_context import LLMCallTracker
from nat.llm.prediction_context import get_call_tracker


def test_tracker_increment():
    tracker = LLMCallTracker()
    assert tracker.increment("func-1") == 1
    assert tracker.increment("func-1") == 2
    assert tracker.increment("func-2") == 1
    assert tracker.increment("func-1") == 3


def test_tracker_reset():
    tracker = LLMCallTracker()
    tracker.increment("func-1")
    tracker.increment("func-1")
    tracker.reset("func-1")
    assert tracker.increment("func-1") == 1


def test_tracker_context_variable():
    tracker1 = get_call_tracker()
    tracker1.increment("func-a")

    tracker2 = get_call_tracker()
    # Should be the same tracker in the same context
    assert tracker2.increment("func-a") == 2
