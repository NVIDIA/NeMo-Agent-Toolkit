# SPDX-FileCopyrightText: Copyright (c) 2025, Your Organization. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import os
from pathlib import Path
from uuid import uuid4
from typing import Optional
import pandas as pd

from pydantic import Field

from aiq.builder.builder import Builder
from aiq.builder.function_info import FunctionInfo
from aiq.cli.register_workflow import register_function
from aiq.data_models.function import FunctionBaseConfig

logger = logging.getLogger(__name__)


class PDFIngestMilvusConfig(FunctionBaseConfig, name="pdf_ingest_milvus"):
    """
    Tool that ingests PDF files into Milvus database with enhanced metadata and ordering.
    Chunks are connected by filename and ordered for sequential retrieval.
    """
    milvus_uri: str = Field(
        default="http://localhost:19530",
        description="Milvus database URI"
    )
    milvus_user: Optional[str] = Field(
        default=None,
        description="Milvus username for authentication"
    )
    milvus_password: Optional[str] = Field(
        default=None,
        description="Milvus password for authentication"
    )
    milvus_db_name: Optional[str] = Field(
        default=None,
        description="Milvus database name"
    )
    collection_name: str = Field(
        default="pdf_manual",
        description="Collection name to store documents"
    )
    embedding_model: str = Field(
        default="nvidia/nv-embedqa-e5-v5",
        description="Embedding model name for NVIDIA NIM"
    )
    chunk_size: int = Field(
        default=1000,
        description="Text chunk size for splitting"
    )
    chunk_overlap: int = Field(
        default=200,
        description="Chunk overlap size"
    )
    clean_file_after_ingest: bool = Field(
        default=False,
        description="Delete PDF file after successful ingestion"
    )


@register_function(config_type=PDFIngestMilvusConfig)
async def pdf_ingest_milvus(config: PDFIngestMilvusConfig, builder: Builder):
    
    async def ingest_pdf_to_milvus(file_path: str) -> str:
        """
        Ingest a PDF file into Milvus database with enhanced metadata and ordering.
        
        Args:
            file_path (str): Path to the PDF file to ingest
            
        Returns:
            str: Success message with ingestion details
        """
        try:
            # Validate file exists
            if not os.path.exists(file_path):
                return f"❌ File not found: {file_path}"
            
            # Validate PDF file
            if not file_path.lower().endswith('.pdf'):
                return f"❌ File must be a PDF: {file_path}"
            
            # Extract filename metadata
            filename = Path(file_path).stem
            full_filename = Path(file_path).name
            file_size = os.path.getsize(file_path)
            
            logger.info(f"Starting PDF ingestion: {filename}")
            
            # Import required libraries
            from langchain_community.document_loaders import PyPDFLoader
            from langchain_text_splitters import RecursiveCharacterTextSplitter
            from langchain_milvus import Milvus
            from langchain_core.documents import Document
            from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
            
            # Set up embedder
            embedder = NVIDIAEmbeddings(
                model=config.embedding_model,
                truncate="END"
            )
            
            # Set up Milvus connection
            connection_args = {"uri": config.milvus_uri}
            if config.milvus_user:
                connection_args["user"] = config.milvus_user
            if config.milvus_password:
                connection_args["password"] = config.milvus_password
            if config.milvus_db_name:
                connection_args["db_name"] = config.milvus_db_name
            
            vector_store = Milvus(
                embedding_function=embedder,
                collection_name=config.collection_name,
                connection_args=connection_args
            )
            
            # Load PDF with page-level tracking
            logger.info(f"Loading PDF: {file_path}")
            loader = PyPDFLoader(file_path)
            documents = loader.load()
            
            if not documents:
                return f"❌ No content found in PDF: {filename}"
            
            logger.info(f"Loaded {len(documents)} pages from {filename}")
            
            # Split documents while preserving page information
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=config.chunk_size,
                chunk_overlap=config.chunk_overlap
            )
            
            all_splits = []
            global_chunk_index = 0
            total_text_length = 0
            
            for page_num, doc in enumerate(documents):
                # Split the current page
                page_splits = text_splitter.split_documents([doc])
                
                # Add enhanced metadata to each chunk
                for local_chunk_index, chunk in enumerate(page_splits):
                    total_text_length += len(chunk.page_content)
                    
                    # Enhanced metadata with ordering and connection info
                    enhanced_metadata = {
                        **chunk.metadata,  # Original metadata (includes page info)
                        "filename": filename,                    # PDF name without extension
                        "full_filename": full_filename,          # PDF name with extension
                        "file_path": file_path,                  # Original file path
                        "file_size_bytes": file_size,            # File size
                        "page_number": page_num + 1,             # 1-indexed page number
                        "chunk_index": global_chunk_index,       # Global chunk sequence
                        "page_chunk_index": local_chunk_index,   # Chunk index within page
                        "total_pages": len(documents),           # Total pages in document
                        "document_type": "pdf",                  # Document type identifier
                        "chunk_size": len(chunk.page_content),   # Individual chunk size
                        "ingestion_timestamp": str(pd.Timestamp.now()),  # Ingestion time
                        "chunk_overlap": config.chunk_overlap,   # Overlap configuration
                        "embedding_model": config.embedding_model,  # Model used
                    }
                    
                    # Create new document with enhanced metadata
                    enhanced_chunk = Document(
                        page_content=chunk.page_content,
                        metadata=enhanced_metadata
                    )
                    
                    all_splits.append(enhanced_chunk)
                    global_chunk_index += 1
            
            logger.info(f"Created {len(all_splits)} chunks from {len(documents)} pages")
            
            # Generate unique IDs and ingest
            ids = [str(uuid4()) for _ in all_splits]
            
            logger.info(f"Ingesting {len(all_splits)} chunks into Milvus collection: {config.collection_name}")
            doc_ids = await vector_store.aadd_documents(documents=all_splits, ids=ids)
            
            logger.info(f"Successfully ingested {len(doc_ids)} chunks")
            
            # Clean up file if requested
            if config.clean_file_after_ingest:
                try:
                    os.remove(file_path)
                    logger.info(f"Cleaned up file: {file_path}")
                except Exception as e:
                    logger.warning(f"Could not clean up file {file_path}: {e}")
            
            # Return success message with details
            success_msg = f"""✅ Successfully ingested PDF: '{filename}'
📄 Pages: {len(documents)}
🔢 Chunks: {len(doc_ids)}
📊 Total text: {total_text_length:,} characters
💾 Collection: {config.collection_name}
🔗 Connection metadata preserved for retrieval ordering"""
            
            return success_msg
            
        except Exception as e:
            logger.exception(f"Error ingesting PDF {file_path}: {e}")
            return f"❌ Error ingesting PDF '{Path(file_path).name}': {str(e)}"

    yield FunctionInfo.from_fn(
        ingest_pdf_to_milvus,
        description="Ingest PDF files into Milvus database with filename metadata and chunk ordering for RAG applications"
    ) 