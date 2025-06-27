# SPDX-FileCopyrightText: Copyright (c) 2025, Your Organization. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Registration module for Summarization tools.
Exports all Summarization functions to be available in AIQ toolkit.
"""

from ....pdf_rag_chatbot.src.pdf_rag_chatbot.summarization import summarization_tool

# Export all functions
__all__ = ["summarization_tool"] 