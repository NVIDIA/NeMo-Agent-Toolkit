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

from nat.eval.evaluator.evaluator_model import EvalInput
from nat.eval.evaluator.evaluator_model import EvalInputItem

logger = logging.getLogger(__name__)


def add_metadata_and_filter(eval_input: EvalInput) -> EvalInput:
    """
    Example custom pre-evaluation process function that:
    1. Adds metadata to each eval input item
    2. Filters out items based on certain criteria
    3. Enriches the full_dataset_entry with additional information

    This function demonstrates how to modify EvalInput objects after
    the workflow has run but before evaluation begins.

    Args:
        eval_input: The EvalInput object to pre-evaluation process

    Returns:
        Modified EvalInput object with additional metadata and filtering applied
    """

    processed_items = []

    for item in eval_input.eval_input_items:
        # Skip items that don't have a generated answer (workflow didn't complete)
        if not item.output_obj:
            logger.info("Skipping item %s - no output generated", item.id)
            continue

        # Add metadata to the full_dataset_entry
        enhanced_entry = item.full_dataset_entry.copy() if item.full_dataset_entry else {}
        enhanced_entry['pre_eval_process_timestamp'] = datetime.now(timezone.utc).isoformat()
        enhanced_entry['pre_eval_process_version'] = "1.0"
        enhanced_entry['has_output'] = bool(item.output_obj)

        # Add additional analysis based on the output
        if isinstance(item.output_obj, str):
            enhanced_entry['output_length'] = len(item.output_obj)
            enhanced_entry['contains_calculation'] = any(op in item.output_obj for op in ['+', '-', '*', '/', '='])

        # Create a new EvalInputItem with the enhanced data
        enhanced_item = EvalInputItem(id=item.id,
                                      input_obj=item.input_obj,
                                      expected_output_obj=item.expected_output_obj,
                                      output_obj=item.output_obj,
                                      trajectory=item.trajectory,
                                      expected_trajectory=item.expected_trajectory,
                                      full_dataset_entry=enhanced_entry)

        processed_items.append(enhanced_item)

    logger.info("Pre-evaluation processing complete: %d items processed from %d original items",
                len(processed_items),
                len(eval_input.eval_input_items))

    return EvalInput(eval_input_items=processed_items)


def normalize_calculator_outputs(eval_input: EvalInput) -> EvalInput:
    """
    Example custom pre-evaluation process function specifically for calculator workflows.
    Normalizes numerical outputs to ensure consistent formatting for evaluation.

    Args:
        eval_input: The EvalInput object to pre-evaluation process

    Returns:
        EvalInput object with normalized numerical outputs
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

    processed_items = []

    for item in eval_input.eval_input_items:
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

        # Create enhanced dataset entry with normalization info
        enhanced_entry = item.full_dataset_entry.copy() if item.full_dataset_entry else {}
        enhanced_entry['output_normalized'] = normalized_output != item.output_obj
        enhanced_entry['expected_normalized'] = normalized_expected != item.expected_output_obj

        # Create new item with normalized values
        normalized_item = EvalInputItem(id=item.id,
                                        input_obj=item.input_obj,
                                        expected_output_obj=normalized_expected,
                                        output_obj=normalized_output,
                                        trajectory=item.trajectory,
                                        expected_trajectory=item.expected_trajectory,
                                        full_dataset_entry=enhanced_entry)

        processed_items.append(normalized_item)

    # Log normalization summary
    output_normalized_count = sum(1 for item in processed_items
                                  if item.full_dataset_entry.get('output_normalized', False))
    expected_normalized_count = sum(1 for item in processed_items
                                    if item.full_dataset_entry.get('expected_normalized', False))

    logger.info("Normalization complete: %d items processed", len(processed_items))
    logger.info("Normalization summary: %d outputs normalized, %d expected outputs normalized",
                output_normalized_count,
                expected_normalized_count)

    return EvalInput(eval_input_items=processed_items)
