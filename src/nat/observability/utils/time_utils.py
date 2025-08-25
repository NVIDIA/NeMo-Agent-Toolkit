# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import logging

logger = logging.getLogger(__name__)


def ns_timestamp(seconds_float: float) -> int:
    """
    Convert a float timestamp in seconds to an integer nanosecond timestamp.

    Args:
        seconds_float (float): The timestamp in seconds (as a float).

    Returns:
        int: The timestamp in nanoseconds (as an integer).
    """
    return int(seconds_float * 1e9)
