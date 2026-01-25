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

"""Answer cleaning utilities for GAIA-style evaluation."""

import re


def clean_answer(response: str | None) -> str | None:
    """Clean and extract the final answer from agent response.

    Removes common prefixes, explanatory text, and formatting to extract
    just the core answer for GAIA-style evaluation.

    Args:
        response: The raw agent response, or None.

    Returns:
        Cleaned answer string, or None if input was None.
    """
    if response is None:
        return None
    if not response:
        return response

    text = response.strip()

    # Common prefixes to remove (case-insensitive)
    prefixes = [
        r"^the\s+(final\s+)?answer\s+is[:\s]+",
        r"^based\s+on\s+(my\s+)?(analysis|research|findings)[,:\s]+",
        r"^after\s+(analyzing|reviewing|examining)[^,]*[,:\s]+",
        r"^therefore[,:\s]+",
        r"^so[,:\s]+",
        r"^in\s+conclusion[,:\s]+",
        r"^to\s+answer\s+(your\s+)?question[,:\s]+",
        r"^the\s+result\s+is[:\s]+",
        r"^i\s+found\s+that[:\s]+",
        r"^my\s+answer\s+is[:\s]+",
    ]

    for pattern in prefixes:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE)

    # Remove trailing explanation in parentheses
    text = re.sub(r"\s*\([^)]*\)\s*$", "", text)

    # Remove trailing punctuation (but keep decimal points in numbers)
    text = text.rstrip(".,;:")

    # If the answer looks like a number, extract just the number
    # Use [^\d-]* to preserve the minus sign for negative numbers
    number_match = re.search(
        r"^[^\d-]*(-?\d+(?:\.\d+)?)\s*"
        r"(?:thousand|million|billion|hours?|minutes?|seconds?|meters?|m\^?\d*)?[^\d]*$",
        text,
        re.IGNORECASE,
    )
    if number_match and len(text) < 100:  # Only for short responses
        # Check if the question expects just a number
        text = number_match.group(1)

    return text.strip()
