# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


def build_oracle_feedback(reasoning_list: list[str], max_chars: int) -> str | None:
    """
    Build truncated feedback string from worst items reasoning.

    Args:
        reasoning_list: List of reasoning strings from worst-performing items.
        max_chars: Maximum characters for the output.

    Returns:
        Formatted feedback string, or None if no reasoning available.
    """
    if not reasoning_list:
        return None

    feedback_parts: list[str] = []
    current_length = 0

    truncated = False
    for i, reasoning in enumerate(reasoning_list, 1):
        entry = f"{i}. {reasoning}\n"
        if current_length + len(entry) > max_chars:
            remaining = max_chars - current_length
            if remaining > 20:  # Only add if meaningful space left
                feedback_parts.append(entry[: remaining - 3] + "...")
            else:
                truncated = True
            break
        feedback_parts.append(entry)
        current_length += len(entry)

    if not feedback_parts:
        return None

    result = "".join(feedback_parts)
    # Add truncation indicator if items were skipped without partial inclusion
    if truncated and not result.endswith("..."):
        # Trim trailing newline if present, add truncation marker
        result = result.rstrip("\n") + "...\n"

    return result
