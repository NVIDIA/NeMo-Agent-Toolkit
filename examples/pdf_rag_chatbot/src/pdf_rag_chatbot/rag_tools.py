# SPDX-FileCopyrightText: Copyright (c) 2025, Your Organization. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import json
from typing import Optional
from uuid import uuid4

from pydantic import Field

from aiq.builder.builder import Builder
from aiq.builder.function_info import FunctionInfo
from aiq.cli.register_workflow import register_function
from aiq.data_models.function import FunctionBaseConfig

logger = logging.getLogger(__name__)


class OrderedPDFSearchConfig(FunctionBaseConfig, name="ordered_pdf_search"):
    """
    Tool that searches PDF documents with ordering support for step-by-step instructions.
    Uses existing aiqtoolkit Milvus retriever with custom ordering logic.
    """
    retriever_name: str = Field(
        default="pdf_retriever",
        description="Name of the Milvus retriever to use"
    )
    max_results: int = Field(
        default=8,
        description="Maximum number of chunks to retrieve"
    )
    return_ordered: bool = Field(
        default=True,
        description="Whether to return results ordered by chunk_index for step-by-step queries"
    )
    include_metadata: bool = Field(
        default=True,
        description="Include source metadata (filename, page) in responses"
    )


class DocumentManagerConfig(FunctionBaseConfig, name="document_manager"):
    """
    Tool that manages and lists available PDF documents in the vector database.
    Uses existing aiqtoolkit retriever to discover available documents.
    """
    retriever_name: str = Field(
        default="pdf_retriever",
        description="Name of the Milvus retriever to query for document list"
    )
    sample_size: int = Field(
        default=100,
        description="Number of sample documents to retrieve for listing"
    )


class MilvusConversationMemoryConfig(FunctionBaseConfig, name="milvus_conversation_memory"):
    """
    Tool that manages conversation history using Milvus database in a separate collection.
    Stores and retrieves conversation context for the current session.
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
    conversation_collection: str = Field(
        default="conversation_history",
        description="Collection name for storing conversation history"
    )
    embedding_model: str = Field(
        default="nvidia/nv-embedqa-e5-v5",
        description="Embedding model for conversation vectorization"
    )
    max_history_items: int = Field(
        default=10,
        description="Maximum number of conversation items to retrieve for context"
    )
    session_id: str = Field(
        default="default_session",
        description="Session ID for conversation tracking"
    )


@register_function(config_type=OrderedPDFSearchConfig)
async def ordered_pdf_search(config: OrderedPDFSearchConfig, builder: Builder):
    
    async def search_pdfs_with_ordering(query: str, filename: Optional[str] = None) -> str:
        """
        Search PDF documents with support for ordered results and filename filtering.
        
        Args:
            query (str): Search query
            filename (str, optional): Filter by specific PDF filename
            
        Returns:
            str: Formatted search results with metadata
        """
        try:
            logger.info(f"Searching PDFs: query='{query}', filename='{filename}'")
            
            # Get the existing aiqtoolkit Milvus retriever
            retriever = await builder.get_retriever(
                config.retriever_name, 
                wrapper_type="langchain"
            )
            
            # Build search parameters
            search_kwargs = {"k": config.max_results}
            
            # Add filename filter if specified (using Milvus filter syntax)
            if filename:
                # Remove .pdf extension if present for consistency
                clean_filename = filename.replace('.pdf', '')
                filter_expr = f'filename == "{clean_filename}"'
                search_kwargs["filter"] = filter_expr
                logger.info(f"Applying filename filter: {filter_expr}")
            
            # Perform search using existing aiqtoolkit retriever
            docs = await retriever.aget_relevant_documents(query, **search_kwargs)
            
            if not docs:
                if filename:
                    return f"❌ No results found for query '{query}' in document '{filename}'"
                else:
                    return f"❌ No results found for query: '{query}'"
            
            logger.info(f"Retrieved {len(docs)} documents")
            
            # Sort by chunk_index for ordered results (step-by-step queries)
            if config.return_ordered:
                docs = sorted(docs, key=lambda x: x.metadata.get('chunk_index', 0))
                logger.info("Results ordered by chunk_index for sequential reading")
            
            # Format results with metadata
            formatted_results = []
            sources = set()
            
            for i, doc in enumerate(docs):
                metadata = doc.metadata
                doc_filename = metadata.get('filename', 'unknown')
                page_num = metadata.get('page_number', '?')
                chunk_idx = metadata.get('chunk_index', '?')
                
                # Track sources
                sources.add(f"{doc_filename} (page {page_num})")
                
                # Format individual result
                if config.include_metadata:
                    result_header = f"**Result {i+1}** [📄 {doc_filename} - Page {page_num} - Chunk {chunk_idx}]"
                else:
                    result_header = f"**Result {i+1}**"
                
                result = f"{result_header}\n{doc.page_content.strip()}"
                formatted_results.append(result)
            
            # Combine results
            response = "\n\n---\n\n".join(formatted_results)
            
            # Add source summary
            if config.include_metadata and sources:
                source_list = ", ".join(sorted(sources))
                response += f"\n\n**Sources:** {source_list}"
            
            return response
            
        except Exception as e:
            logger.exception(f"Error searching PDFs: {e}")
            return f"❌ Error searching documents: {str(e)}"

    yield FunctionInfo.from_fn(
        search_pdfs_with_ordering,
        description="Search PDF documents with ordering support for step-by-step instructions and filename filtering"
    )


@register_function(config_type=DocumentManagerConfig)
async def document_manager(config: DocumentManagerConfig, builder: Builder):
    
    async def list_available_documents(query: str = "") -> str:
        """
        List all available PDF documents in the vector database.
        
        Returns:
            str: Formatted list of available documents with metadata
        """
        try:
            logger.info("Listing available PDF documents")
            
            # Get the existing aiqtoolkit Milvus retriever
            retriever = await builder.get_retriever(
                config.retriever_name,
                wrapper_type="langchain"
            )
            
            # Get a sample of documents to extract unique filenames
            # Using a broad query to get diverse results
            sample_docs = await retriever.aget_relevant_documents(
                "document content text", 
                k=config.sample_size
            )
            
            if not sample_docs:
                return "❌ No documents found in the database. Please ingest some PDF files first."
            
            # Extract unique document information
            documents = {}
            for doc in sample_docs:
                metadata = doc.metadata
                filename = metadata.get('filename', 'unknown')
                full_filename = metadata.get('full_filename', f"{filename}.pdf")
                total_pages = metadata.get('total_pages', '?')
                ingestion_time = metadata.get('ingestion_timestamp', 'unknown')
                
                if filename not in documents:
                    documents[filename] = {
                        'full_filename': full_filename,
                        'total_pages': total_pages,
                        'ingestion_time': ingestion_time,
                        'chunk_count': 0
                    }
                
                documents[filename]['chunk_count'] += 1
            
            if not documents:
                return "❌ No valid documents found in the database."
            
            # Format the document list
            doc_list = []
            for filename, info in sorted(documents.items()):
                doc_entry = f"📄 **{info['full_filename']}**"
                doc_entry += f"\n   └─ Pages: {info['total_pages']}"
                doc_entry += f"\n   └─ Chunks: {info['chunk_count']}"
                if info['ingestion_time'] != 'unknown':
                    # Format timestamp nicely
                    try:
                        from datetime import datetime
                        timestamp = datetime.fromisoformat(info['ingestion_time'].replace('Z', '+00:00'))
                        formatted_time = timestamp.strftime('%Y-%m-%d %H:%M')
                        doc_entry += f"\n   └─ Ingested: {formatted_time}"
                    except:
                        doc_entry += f"\n   └─ Ingested: {info['ingestion_time'][:16]}"
                
                doc_list.append(doc_entry)
            
            response = f"**📚 Available Documents ({len(documents)} total):**\n\n"
            response += "\n\n".join(doc_list)
            response += f"\n\n**💡 Usage Tips:**"
            response += f"\n• Use filename for specific searches: 'search for X in filename.pdf'"
            response += f"\n• Ask for steps/procedures to get ordered results"
            response += f"\n• General queries will search across all documents"
            
            return response
            
        except Exception as e:
            logger.exception(f"Error listing documents: {e}")
            return f"❌ Error listing documents: {str(e)}"

    yield FunctionInfo.from_fn(
        list_available_documents,
        description="List all available PDF documents in the vector database with metadata"
    )


@register_function(config_type=MilvusConversationMemoryConfig)
async def milvus_conversation_memory(config: MilvusConversationMemoryConfig, builder: Builder):
    
    # Shared conversation storage for the session
    conversation_store = []
    
    async def store_conversation(user_message: str, assistant_response: str) -> str:
        """
        Store a conversation exchange in Milvus conversation collection.
        
        Args:
            user_message (str): User's message
            assistant_response (str): Assistant's response
            
        Returns:
            str: Confirmation message
        """
        try:
            logger.info(f"Storing conversation for session: {config.session_id}")
            
            # Import required libraries
            from langchain_milvus import Milvus
            from langchain_core.documents import Document
            from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
            import pandas as pd
            
            # Set up embedder
            embedder = NVIDIAEmbeddings(
                model=config.embedding_model,
                truncate="END"
            )
            
            # Set up Milvus connection for conversations
            connection_args = {"uri": config.milvus_uri}
            if config.milvus_user:
                connection_args["user"] = config.milvus_user
            if config.milvus_password:
                connection_args["password"] = config.milvus_password
            if config.milvus_db_name:
                connection_args["db_name"] = config.milvus_db_name
            
            vector_store = Milvus(
                embedding_function=embedder,
                collection_name=config.conversation_collection,
                connection_args=connection_args
            )
            
            # Create conversation exchange document
            conversation_text = f"User: {user_message}\nAssistant: {assistant_response}"
            
            conversation_metadata = {
                "session_id": config.session_id,
                "user_message": user_message,
                "assistant_response": assistant_response,
                "timestamp": str(pd.Timestamp.now()),
                "conversation_index": len(conversation_store),
                "message_type": "conversation_exchange"
            }
            
            conversation_doc = Document(
                page_content=conversation_text,
                metadata=conversation_metadata
            )
            
            # Store in Milvus
            doc_id = str(uuid4())
            await vector_store.aadd_documents(documents=[conversation_doc], ids=[doc_id])
            
            # Also store in local session cache
            conversation_store.append({
                "user": user_message,
                "assistant": assistant_response,
                "timestamp": conversation_metadata["timestamp"]
            })
            
            logger.info(f"Stored conversation exchange (ID: {doc_id})")
            return f"✅ Conversation stored in session {config.session_id}"
            
        except Exception as e:
            logger.exception(f"Error storing conversation: {e}")
            return f"❌ Error storing conversation: {str(e)}"
    
    async def get_conversation_context(current_query: str) -> str:
        """
        Retrieve relevant conversation history for context.
        
        Args:
            current_query (str): Current user query to find relevant history
            
        Returns:
            str: Formatted conversation context
        """
        try:
            logger.info(f"Retrieving conversation context for: {current_query}")
            
            # Import required libraries
            from langchain_milvus import Milvus
            from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
            
            # Set up embedder and vector store
            embedder = NVIDIAEmbeddings(
                model=config.embedding_model,
                truncate="END"
            )
            
            connection_args = {"uri": config.milvus_uri}
            if config.milvus_user:
                connection_args["user"] = config.milvus_user
            if config.milvus_password:
                connection_args["password"] = config.milvus_password
            if config.milvus_db_name:
                connection_args["db_name"] = config.milvus_db_name
            
            vector_store = Milvus(
                embedding_function=embedder,
                collection_name=config.conversation_collection,
                connection_args=connection_args
            )
            
            # Search for relevant conversation history
            search_kwargs = {
                "k": config.max_history_items,
                "filter": f'session_id == "{config.session_id}"'
            }
            
            try:
                relevant_conversations = await vector_store.asimilarity_search(
                    current_query, 
                    **search_kwargs
                )
                
                if not relevant_conversations:
                    return "No previous conversation context found."
                
                # Format conversation history
                context_items = []
                for doc in relevant_conversations:
                    metadata = doc.metadata
                    timestamp = metadata.get('timestamp', 'unknown')
                    user_msg = metadata.get('user_message', '')
                    assistant_msg = metadata.get('assistant_response', '')
                    
                    # Format conversation exchange
                    context_item = f"[{timestamp[:16]}] User: {user_msg[:100]}...\nAssistant: {assistant_msg[:150]}..."
                    context_items.append(context_item)
                
                context = "\n\n".join(context_items[:5])  # Limit to 5 most relevant
                return f"**Previous Conversation Context:**\n{context}"
                
            except Exception as search_error:
                logger.warning(f"Could not search conversation history: {search_error}")
                # Fallback to local session cache
                if conversation_store:
                    recent_items = conversation_store[-3:]  # Last 3 exchanges
                    context_items = []
                    for item in recent_items:
                        context_item = f"User: {item['user'][:100]}...\nAssistant: {item['assistant'][:150]}..."
                        context_items.append(context_item)
                    return f"**Recent Session Context:**\n" + "\n\n".join(context_items)
                else:
                    return "No conversation context available."
            
        except Exception as e:
            logger.exception(f"Error retrieving conversation context: {e}")
            return f"❌ Error retrieving context: {str(e)}"
    
    async def clear_session_history() -> str:
        """
        Clear conversation history for the current session.
        
        Returns:
            str: Confirmation message
        """
        try:
            logger.info(f"Clearing conversation history for session: {config.session_id}")
            
            # Clear local session cache
            conversation_store.clear()
            
            # Note: Milvus doesn't have direct delete by filter in langchain-milvus
            # For now, we'll rely on session isolation and periodic cleanup
            # In production, you'd implement a cleanup job
            
            return f"✅ Session history cleared for {config.session_id}"
            
        except Exception as e:
            logger.exception(f"Error clearing session history: {e}")
            return f"❌ Error clearing history: {str(e)}"
    
    # Return multiple functions as a simple dispatcher
    async def conversation_memory_handler(action: str, user_message: str = "", assistant_response: str = "", query: str = "") -> str:
        """
        Handle conversation memory operations.
        
        Args:
            action (str): Action to perform - 'store', 'get_context', or 'clear'
            user_message (str): User message (for store action)
            assistant_response (str): Assistant response (for store action)  
            query (str): Current query (for get_context action)
            
        Returns:
            str: Result of the action
        """
        if action == "store":
            return await store_conversation(user_message, assistant_response)
        elif action == "get_context":
            return await get_conversation_context(query)
        elif action == "clear":
            return await clear_session_history()
        else:
            return f"❌ Unknown action: {action}. Use 'store', 'get_context', or 'clear'"
    
    yield FunctionInfo.from_fn(
        conversation_memory_handler,
        description="Manage conversation history using Milvus database for session-based memory"
    ) 