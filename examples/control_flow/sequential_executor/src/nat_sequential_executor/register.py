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

import json
import logging
import re

from nat.builder.builder import Builder
from nat.builder.framework_enum import LLMFrameworkEnum
from nat.builder.function_info import FunctionInfo
from nat.cli.register_workflow import register_function
from nat.data_models.function import FunctionBaseConfig

logger = logging.getLogger(__name__)


class TextProcessorFunctionConfig(FunctionBaseConfig, name="text_processor"):
    """Configuration for the text processor function."""
    pass


@register_function(config_type=TextProcessorFunctionConfig, framework_wrappers=[LLMFrameworkEnum.LANGCHAIN])
async def text_processor_function(config: TextProcessorFunctionConfig, builder: Builder):
    """
    Create a text processor function that cleans and processes raw text input.
    This is the first step in a data processing pipeline.

    Parameters
    ----------
    config : TextProcessorFunctionConfig
        Configuration for the text processor function
    builder : Builder
        The NeMo Agent toolkit builder instance

    Returns
    -------
    A FunctionInfo object that processes raw text and returns structured data
    """

    async def process_text(raw_text: str) -> str:
        """
        Process raw text by cleaning and extracting key information.

        Args:
            raw_text: Raw text input to process

        Returns:
            JSON string containing processed text data
        """
        # Clean the text by removing extra whitespace and special characters
        cleaned_text = re.sub(r'\s+', ' ', raw_text.strip())
        cleaned_text = re.sub(r'[^\w\s\.\,\!\?]', '', cleaned_text)

        # Extract basic statistics
        word_count = len(cleaned_text.split())
        sentence_count = len(re.findall(r'[.!?]+', cleaned_text))

        # Create structured output
        processed_data = {
            "cleaned_text": cleaned_text,
            "word_count": word_count,
            "sentence_count": sentence_count,
            "processing_status": "completed"
        }

        return json.dumps(processed_data)

    yield FunctionInfo.from_fn(process_text, description="Process raw text by cleaning and extracting basic statistics")


class DataAnalyzerFunctionConfig(FunctionBaseConfig, name="data_analyzer"):
    """Configuration for the data analyzer function."""
    pass


@register_function(config_type=DataAnalyzerFunctionConfig, framework_wrappers=[LLMFrameworkEnum.LANGCHAIN])
async def data_analyzer_function(config: DataAnalyzerFunctionConfig, builder: Builder):
    """
    Create a data analyzer function that analyzes processed text data.
    This is the second step in a data processing pipeline.

    Parameters
    ----------
    config : DataAnalyzerFunctionConfig
        Configuration for the data analyzer function
    builder : Builder
        The NeMo Agent toolkit builder instance

    Returns
    -------
    A FunctionInfo object that analyzes text data and returns insights
    """

    async def analyze_data(processed_data: str) -> str:
        """
        Analyze processed text data and generate insights.

        Args:
            processed_data: JSON string containing processed text data

        Returns:
            JSON string containing analysis results
        """
        try:
            # Parse the input data
            data = json.loads(processed_data)

            # Perform analysis
            text = data.get("cleaned_text", "")
            word_count = data.get("word_count", 0)
            sentence_count = data.get("sentence_count", 0)

            # Calculate metrics
            avg_words_per_sentence = word_count / sentence_count if sentence_count > 0 else 0

            # Determine text complexity
            if avg_words_per_sentence < 10:
                complexity = "simple"
            elif avg_words_per_sentence < 20:
                complexity = "moderate"
            else:
                complexity = "complex"

            # Find most common words (simple implementation)
            words = text.lower().split()
            word_freq = {}
            for word in words:
                if len(word) > 3:  # Only count words longer than 3 characters
                    word_freq[word] = word_freq.get(word, 0) + 1

            top_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:5]

            # Create analysis results
            analysis_results = {
                "word_count": word_count,
                "sentence_count": sentence_count,
                "avg_words_per_sentence": round(avg_words_per_sentence, 2),
                "complexity": complexity,
                "top_words": [word for word, count in top_words],
                "analysis_status": "completed"
            }

            return json.dumps(analysis_results)

        except json.JSONDecodeError:
            # Handle invalid JSON input
            error_result = {"error": "Invalid input format", "analysis_status": "failed"}
            return json.dumps(error_result)

    yield FunctionInfo.from_fn(
        analyze_data, description="Analyze processed text data and generate insights about complexity and content")


class ReportGeneratorFunctionConfig(FunctionBaseConfig, name="report_generator"):
    """Configuration for the report generator function."""
    pass


@register_function(config_type=ReportGeneratorFunctionConfig, framework_wrappers=[LLMFrameworkEnum.LANGCHAIN])
async def report_generator_function(config: ReportGeneratorFunctionConfig, builder: Builder):
    """
    Create a report generator function that creates a summary report from analysis data.
    This is the final step in a data processing pipeline.

    Parameters
    ----------
    config : ReportGeneratorFunctionConfig
        Configuration for the report generator function
    builder : Builder
        The NeMo Agent toolkit builder instance

    Returns
    -------
    A FunctionInfo object that generates a formatted report from analysis data
    """

    async def generate_report(analysis_data: str) -> str:
        """
        Generate a formatted report from analysis data.

        Args:
            analysis_data: JSON string containing analysis results

        Returns:
            Formatted text report
        """
        try:
            # Parse the analysis data
            data = json.loads(analysis_data)

            if data.get("analysis_status") == "failed":
                return "Report Generation Failed: " + data.get("error", "Unknown error")

            # Generate the report
            report_lines = [
                "=== TEXT ANALYSIS REPORT ===",
                "",
                "Text Statistics:",
                f"  - Word Count: {data.get('word_count', 0)}",
                f"  - Sentence Count: {data.get('sentence_count', 0)}",
                f"  - Average Words per Sentence: {data.get('avg_words_per_sentence', 0)}",
                f"  - Text Complexity: {data.get('complexity', 'unknown').title()}",
                "",
                "Top Words:",
            ]

            # Add top words to report
            top_words = data.get('top_words', [])
            if top_words:
                for i, word in enumerate(top_words, 1):
                    report_lines.append(f"  {i}. {word}")
            else:
                report_lines.append("  No significant words found")

            report_lines.extend(["", "Report generated successfully.", "=========================="])

            return "\n".join(report_lines)

        except json.JSONDecodeError:
            return "Report Generation Failed: Invalid analysis data format"

    yield FunctionInfo.from_fn(generate_report,
                               description="Generate a formatted text analysis report from analysis data")
