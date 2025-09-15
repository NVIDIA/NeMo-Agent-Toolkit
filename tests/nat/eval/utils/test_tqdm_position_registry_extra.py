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

import pytest

from nat.eval.utils.tqdm_position_registry import TqdmPositionRegistry


def test_claim_and_release_positions():
    pos = TqdmPositionRegistry.claim()
    assert isinstance(pos, int)
    TqdmPositionRegistry.release(pos)
    # after release, we should be able to claim the same position again quickly
    reclaimed = TqdmPositionRegistry.claim()
    TqdmPositionRegistry.release(reclaimed)


def test_exhaust_positions_then_error(monkeypatch):
    # set small max to speed up
    monkeypatch.setattr(TqdmPositionRegistry, "_max_positions", 2)
    # reset positions
    # Reset positions (test-only)
    TqdmPositionRegistry._positions.clear()
    a = TqdmPositionRegistry.claim()
    b = TqdmPositionRegistry.claim()
    assert {a, b} == {0, 1}
    with pytest.raises(RuntimeError):
        TqdmPositionRegistry.claim()
    # cleanup
    TqdmPositionRegistry.release(a)
    TqdmPositionRegistry.release(b)
