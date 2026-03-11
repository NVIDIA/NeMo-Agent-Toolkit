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

import json
import os
import tempfile

from nemo_evaluator.contrib.byob import ScorerInput
from nemo_evaluator.contrib.byob import benchmark
from nemo_evaluator.contrib.byob import scorer
from nemo_evaluator.contrib.byob.scorers import exact_match

# Create a tiny test dataset as a temp file
_DATA = [
    {
        "id": "0", "question": "What is 2+2?", "target": "4"
    },
    {
        "id": "1", "question": "What color is the sky?", "target": "blue"
    },
    {
        "id": "2", "question": "Capital of France?", "target": "Paris"
    },
]

_DATASET_PATH = os.path.join(tempfile.gettempdir(), "byob_test_dataset.jsonl")
with open(_DATASET_PATH, "w") as f:
    for row in _DATA:
        f.write(json.dumps(row) + "\n")


@benchmark(
    name="test-exact-match",
    dataset=_DATASET_PATH,
    prompt="{question}",
    target_field="target",
)
@scorer
def test_exact_match_scorer(sample: ScorerInput) -> dict:
    return exact_match(sample)
