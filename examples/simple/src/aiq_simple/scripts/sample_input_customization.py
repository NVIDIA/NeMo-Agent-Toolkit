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
"""
Sample input custom script to convert the input to a JSON file that can be used
by the aiq eval system.
"""

import argparse
import json
from pathlib import Path


def transform_record(raw: dict, index: int) -> dict:
    """
    Transform one record from original to final format.
    """
    return {
        "id": str(index + 1),  # reindex as "1", "2", ...
        "question": raw["content"]["question_text"].strip(),
        "answer": raw["content"]["answer_text"].strip()
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_path", type=Path, required=True)
    parser.add_argument("--output_path", type=Path, required=True)
    args = parser.parse_args()

    with args.file_path.open() as infile:
        original_data = json.load(infile)

    transformed_data = [transform_record(entry, i) for i, entry in enumerate(original_data)]

    with args.output_path.open("w") as outfile:
        json.dump(transformed_data, outfile, indent=2)


if __name__ == "__main__":
    main()
