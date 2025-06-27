# SPDX-FileCopyrightText: Copyright (c) 2025, Your Organization. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Text File Summarization module.
Provides functionality for reading text files and generating summaries.
"""

import logging
import os
from pathlib import Path
from typing import Dict, Any, Optional, List
import re

from pydantic import Field

from aiq.builder.builder import Builder
from aiq.builder.function_info import FunctionInfo
from aiq.cli.register_workflow import register_function
from aiq.data_models.function import FunctionBaseConfig

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class SummarizationConfig(FunctionBaseConfig, name="summarization_tool"):
    """Configuration for Text File Summarization tool."""
    max_chunk_size: int = Field(default=4000, description="Maximum size of text chunks for processing")
    summary_length: int = Field(default=200, description="Target length of summary in words")
    include_metadata: bool = Field(default=True, description="Include file metadata in results")
    supported_extensions: List[str] = Field(
        default=[".txt", ".md", ".rst", ".py", ".js", ".html", ".css", ".json", ".xml", ".csv"],
        description="List of supported file extensions"
    )

def _read_text_file(file_path: str) -> Dict[str, Any]:
    """Read text content from a file."""
    try:
        path = Path(file_path)
        
        if not path.exists():
            return {"status": "error", "error": f"File not found: {file_path}"}
        
        if not path.is_file():
            return {"status": "error", "error": f"Path is not a file: {file_path}"}
        
        # Check file extension
        config = SummarizationConfig()
        if path.suffix.lower() not in config.supported_extensions:
            return {
                "status": "error", 
                "error": f"Unsupported file type: {path.suffix}. Supported types: {config.supported_extensions}"
            }
        
        # Read file content
        with open(path, 'r', encoding='utf-8') as file:
            content = file.read()
        
        # Get file metadata
        metadata = {
            "file_name": path.name,
            "file_path": str(path.absolute()),
            "file_size": path.stat().st_size,
            "file_extension": path.suffix.lower(),
            "last_modified": path.stat().st_mtime,
        }
        
        return {
            "status": "success",
            "content": content,
            "metadata": metadata,
            "word_count": len(content.split()),
            "character_count": len(content)
        }
        
    except UnicodeDecodeError:
        return {"status": "error", "error": f"Unable to decode file as UTF-8: {file_path}"}
    except Exception as e:
        logger.exception(f"Error reading file {file_path}: {e}")
        return {"status": "error", "error": str(e)}

def _chunk_text(text: str, max_chunk_size: int) -> List[str]:
    """Split text into chunks of specified size."""
    words = text.split()
    chunks = []
    current_chunk = []
    current_size = 0
    
    for word in words:
        word_size = len(word) + 1  # +1 for space
        if current_size + word_size > max_chunk_size and current_chunk:
            chunks.append(" ".join(current_chunk))
            current_chunk = [word]
            current_size = word_size
        else:
            current_chunk.append(word)
            current_size += word_size
    
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    
    return chunks

def _generate_summary(text: str, target_length: int = 200) -> str:
    """Generate a summary of the given text."""
    # Simple extractive summarization using sentence scoring
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    if not sentences:
        return "No content to summarize."
    
    if len(sentences) <= 3:
        return text[:target_length * 5] + "..." if len(text) > target_length * 5 else text
    
    # Score sentences based on word frequency
    words = re.findall(r'\b\w+\b', text.lower())
    word_freq = {}
    for word in words:
        if len(word) > 3:  # Skip short words
            word_freq[word] = word_freq.get(word, 0) + 1
    
    # Score each sentence
    sentence_scores = []
    for sentence in sentences:
        sentence_words = re.findall(r'\b\w+\b', sentence.lower())
        score = sum(word_freq.get(word, 0) for word in sentence_words if len(word) > 3)
        sentence_scores.append((score, sentence))
    
    # Sort by score and take top sentences
    sentence_scores.sort(reverse=True)
    
    # Calculate how many sentences to include
    total_words = sum(len(s.split()) for _, s in sentence_scores[:len(sentence_scores)//2])
    selected_sentences = []
    current_length = 0
    
    for _, sentence in sentence_scores:
        sentence_words = len(sentence.split())
        if current_length + sentence_words <= target_length:
            selected_sentences.append(sentence)
            current_length += sentence_words
        else:
            break
    
    # Sort selected sentences by original order
    original_order = {s: i for i, s in enumerate(sentences)}
    selected_sentences.sort(key=lambda s: original_order.get(s, 0))
    
    summary = ". ".join(selected_sentences)
    if summary and not summary.endswith(('.', '!', '?')):
        summary += "."
    
    return summary

@register_function(config_type=SummarizationConfig)
async def summarization_tool(config: SummarizationConfig, builder: Builder):
    """Register the Text File Summarization tool."""

    async def read_text_file(file_path: str) -> Dict[str, Any]:
        """Read and return the content of a text file."""
        return _read_text_file(file_path)

    async def summarize_text_file(file_path: str) -> Dict[str, Any]:
        """Read a text file and generate a summary."""
        file_result = _read_text_file(file_path)
        
        if file_result["status"] != "success":
            return file_result
        
        content = file_result["content"]
        metadata = file_result.get("metadata", {})
        
        # Generate summary
        summary = _generate_summary(content, config.summary_length)
        
        result = {
            "status": "success",
            "file_path": file_path,
            "summary": summary,
            "summary_length": len(summary.split()),
            "original_length": file_result["word_count"],
            "compression_ratio": round(len(summary.split()) / file_result["word_count"] * 100, 2)
        }
        
        if config.include_metadata:
            result["metadata"] = metadata
        
        return result

    async def summarize_text_content(text_content: str) -> Dict[str, Any]:
        """Generate a summary from provided text content."""
        if not text_content.strip():
            return {"status": "error", "error": "No text content provided"}
        
        summary = _generate_summary(text_content, config.summary_length)
        
        return {
            "status": "success",
            "summary": summary,
            "summary_length": len(summary.split()),
            "original_length": len(text_content.split()),
            "compression_ratio": round(len(summary.split()) / len(text_content.split()) * 100, 2)
        }

    async def chunk_and_summarize_file(file_path: str) -> Dict[str, Any]:
        """Read a large text file, chunk it, and generate summaries for each chunk."""
        file_result = _read_text_file(file_path)
        
        if file_result["status"] != "success":
            return file_result
        
        content = file_result["content"]
        metadata = file_result.get("metadata", {})
        
        # Chunk the text
        chunks = _chunk_text(content, config.max_chunk_size)
        
        # Generate summaries for each chunk
        chunk_summaries = []
        for i, chunk in enumerate(chunks):
            summary = _generate_summary(chunk, config.summary_length // len(chunks))
            chunk_summaries.append({
                "chunk_id": i + 1,
                "chunk_size": len(chunk.split()),
                "summary": summary,
                "summary_length": len(summary.split())
            })
        
        # Generate overall summary
        overall_summary = _generate_summary(content, config.summary_length)
        
        result = {
            "status": "success",
            "file_path": file_path,
            "overall_summary": overall_summary,
            "chunks_processed": len(chunks),
            "chunk_summaries": chunk_summaries,
            "total_original_length": file_result["word_count"],
            "total_summary_length": len(overall_summary.split())
        }
        
        if config.include_metadata:
            result["metadata"] = metadata
        
        return result

    async def list_supported_formats() -> Dict[str, Any]:
        """Return list of supported file formats."""
        return {
            "status": "success",
            "supported_extensions": config.supported_extensions,
            "description": "File extensions that can be processed by the summarization tool"
        }

    # Return all functions as a single FunctionInfo
    yield FunctionInfo.from_fn(
        summarize_text_file,
        description="Reads text files and generates summaries. Supports multiple file formats including .txt, .md, .py, .js, .html, .json, .xml, .csv and more. Returns summary, metadata, and compression statistics."
    ) 