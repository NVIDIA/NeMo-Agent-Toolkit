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

import re

_THINK_BLOCK_PATTERN = re.compile(r'<think>.*?</think>\s*', re.DOTALL)


def remove_r1_think_tags(text: str):
    """Remove every ``<think>...</think>`` block from ``text``.

    Blocks are removed where they appear. The previous ``re.match`` plus a single lazy
    ``.*?</think>`` returned only the text that followed the *first* closing tag, which
    (a) discarded anything the model emitted before that tag, (b) left any later block
    in place, and (c) truncated the text at a ``</think>`` that was never opened.
    """
    return _THINK_BLOCK_PATTERN.sub('', text)
