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
    """Remove reasoning-model think blocks from ``text``.

    Complete ``<think>...</think>`` blocks are removed wherever they appear, so text before,
    between or after them survives. The previous ``re.match`` plus a single lazy
    ``.*?</think>`` returned only what followed the *first* closing tag, which discarded
    anything the model emitted before that tag and left any later block in place.

    A lone ``</think>`` with no opening tag is the provider artifact tracked in #1611.
    Everything before it is unusable preamble, so only what follows the last one is kept.
    That is what lets an all-reasoning message reduce to an empty string and reach the
    ReAct retry path instead of being accepted as a final answer.
    """
    stripped = _THINK_BLOCK_PATTERN.sub('', text)
    if '</think>' in stripped:
        stripped = stripped.rsplit('</think>', 1)[1].lstrip()
    return stripped
