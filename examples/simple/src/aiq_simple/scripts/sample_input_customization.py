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
Sample input custom script to convert input to a JSON or CSV file
that can be used by the AIQ evaluation system.
"""

import argparse
import csv
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


def transform_dataset(input_path: Path,
                      input_format: str,
                      output_path: Path,
                      output_format: str,
                      max_rows: int | None = None):
    """
    Transform the input dataset to a JSON or CSV file that can be used by the AIQ evaluation system.
    """
    if input_format == "json":
        with input_path.open() as infile:
            original_data = json.load(infile)
    elif input_format == "csv":
        with input_path.open() as infile:
            reader = csv.DictReader(infile)
            original_data = list(reader)
    else:
        raise ValueError(f"Unsupported input format: {input_format}")

    if not isinstance(original_data, list):
        raise ValueError("Expected input JSON to be a list of records")

    sliced_data = original_data[:max_rows] if max_rows else original_data
    transformed_data = [transform_record(entry, i) for i, entry in enumerate(sliced_data)]

    if output_format == "json":
        with output_path.open("w") as outfile:
            json.dump(transformed_data, outfile, indent=2)

    elif output_format == "csv":
        fieldnames = ["id", "question", "answer"]
        with output_path.open("w", newline="") as outfile:
            writer = csv.DictWriter(outfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(transformed_data)

    print(f"✅ Transformed {len(transformed_data)} records to {output_format.upper()} at {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=Path, required=True, help="Path to original input JSON")
    parser.add_argument("--input_format",
                        type=str,
                        required=True,
                        choices=["json", "csv"],
                        help="Input format: json or csv")
    parser.add_argument("--output_path", type=Path, required=True, help="Path to transformed output file")
    parser.add_argument("--output_format",
                        type=str,
                        required=True,
                        choices=["json", "csv"],
                        help="Output format: json or csv")
    parser.add_argument("--max_rows", type=int, default=None, help="Maximum number of records to process")
    args = parser.parse_args()
    transform_dataset(args.input_path, args.input_format, args.output_path, args.output_format, args.max_rows)


if __name__ == "__main__":
    main()
