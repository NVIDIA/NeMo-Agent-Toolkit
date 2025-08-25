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

import logging
from datetime import datetime
from datetime import timezone

from nat.eval.evaluator.evaluator_model import EvalInputItem

logger = logging.getLogger(__name__)


def add_metadata_and_filter(item: EvalInputItem) -> EvalInputItem:
    """
    Example custom pre-evaluation process function that:
    1. Adds metadata to the eval input item
    2. Enriches the full_dataset_entry with additional information

    This function demonstrates how to modify individual EvalInputItem objects after
    the workflow has run but before evaluation begins.

    Args:
        item: The EvalInputItem object to pre-evaluation process

    Returns:
        Modified EvalInputItem object with additional metadata applied
    """

    # Skip items that don't have a generated answer (workflow didn't complete)
    if not item.output_obj:
        logger.info("Skipping item %s - no output generated", item.id)
        return item  # Return unchanged item

    # Add metadata to the full_dataset_entry
    enhanced_entry = item.full_dataset_entry.copy() if item.full_dataset_entry else {}
    enhanced_entry['pre_eval_process_timestamp'] = datetime.now(timezone.utc).isoformat()
    enhanced_entry['pre_eval_process_version'] = "1.0"
    enhanced_entry['has_output'] = bool(item.output_obj)

    # Add additional analysis based on the output
    if isinstance(item.output_obj, str):
        enhanced_entry['output_length'] = len(item.output_obj)
        enhanced_entry['contains_calculation'] = any(op in item.output_obj for op in ['+', '-', '*', '/', '='])

    # Return enhanced item
    return item.copy_with_updates(full_dataset_entry=enhanced_entry)


def normalize_calculator_outputs(item: EvalInputItem) -> EvalInputItem:
    """
    Example custom pre-evaluation process function specifically for calculator workflows.
    Normalizes numerical outputs to ensure consistent formatting for evaluation.

    Args:
        item: The EvalInputItem object to pre-evaluation process

    Returns:
        EvalInputItem object with normalized numerical outputs
    """

    def normalize_number(text: str) -> str:
        """Helper function to normalize numerical representations"""
        import re

        # Extract numbers from text and normalize them
        number_pattern = r'-?\d+(?:\.\d+)?'
        numbers = re.findall(number_pattern, text)

        normalized_text = text
        for num_str in numbers:
            try:
                # Convert to float and back to remove unnecessary decimals
                num = float(num_str)
                if num.is_integer():
                    normalized_num = str(int(num))
                else:
                    normalized_num = f"{num:.2f}".rstrip('0').rstrip('.')
                normalized_text = normalized_text.replace(num_str, normalized_num, 1)
            except ValueError:
                continue

        return normalized_text

    # Normalize the output if it exists
    normalized_output = item.output_obj
    if isinstance(item.output_obj, str):
        normalized_output = normalize_number(item.output_obj)
        if normalized_output != item.output_obj:
            logger.info("Item %s - Output normalized: '%s' → '%s'", item.id, item.output_obj, normalized_output)

    # Also normalize the expected output for consistency
    normalized_expected = item.expected_output_obj
    if isinstance(item.expected_output_obj, str):
        normalized_expected = normalize_number(item.expected_output_obj)
        if normalized_expected != item.expected_output_obj:
            logger.info("Item %s - Expected output normalized: '%s' → '%s'",
                        item.id,
                        item.expected_output_obj,
                        normalized_expected)

    # Return item with normalized values (keeping everything else unchanged)
    return item.copy_with_updates(output_obj=normalized_output, expected_output_obj=normalized_expected)
