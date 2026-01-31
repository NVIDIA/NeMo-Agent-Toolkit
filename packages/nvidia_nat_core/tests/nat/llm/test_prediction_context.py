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
