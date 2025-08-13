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


class TqdmPositionRegistry:
    """
    A simple registry for tqdm positions.
    """
    _positions = set()
    _max_positions = 100

    @classmethod
    def claim(cls) -> int:
        """
        Claim a tqdm position in the range of 0-99.
        """
        for i in range(cls._max_positions):
            if i not in cls._positions:
                cls._positions.add(i)
                return i
        raise RuntimeError("No available tqdm positions.")

    @classmethod
    def release(cls, pos: int):
        """
        Release a tqdm position.
        """
        cls._positions.discard(pos)
