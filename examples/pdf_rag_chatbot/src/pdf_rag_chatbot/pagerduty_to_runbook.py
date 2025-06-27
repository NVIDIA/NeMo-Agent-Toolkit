import logging
import re
import os
from typing import Optional, Dict, Any, List
from uuid import uuid4

import httpx
from pydantic import Field

from aiq.builder.builder import Builder
from aiq.builder.function_info import FunctionInfo
from aiq.cli.register_workflow import register_function
from aiq.data_models.function import FunctionBaseConfig

logger = logging.getLogger(__name__)


class PagerDutyToRunbookConfig(FunctionBaseConfig, name="pagerduty_to_runbook"):
    """
    Function that extracts alert name from PagerDuty incident and returns complete matching PDF runbook.
    Eliminates LLM inconsistencies by doing direct Milvus queries.
    """
    # PagerDuty API settings
    pagerduty_api_token: Optional[str] = Field(
        default=None,
        description="PagerDuty API token for authentication"
    )
    pagerduty_api_version: str = Field(
        default="2",
        description="PagerDuty API version"
    )
    pagerduty_timeout: int = Field(
        default=30,
        description="Timeout for PagerDuty API calls"
    )
    
    # Milvus settings
    milvus_uri: str = Field(
        default="https://milvus-playground-dgxc-sre-blr-dev.db.nvw.nvidia.com",
        description="Milvus database URI"
    )
    milvus_user: str = Field(
        default="dgxc_sre_blr_owner",
        description="Milvus username"
    )
    milvus_password: str = Field(
        default="oc1AMM8RgTyN7FqT",
        description="Milvus password"
    )
    milvus_db_name: str = Field(
        default="dgxc_sre_blr",
        description="Milvus database name"
    )
    collection_name: str = Field(
        default="pdf_documents",
        description="Milvus collection name for PDF documents"
    )
    embedding_model: str = Field(
        default="nvidia/nv-embedqa-e5-v5",
        description="Embedding model for search"
    )


@register_function(config_type=PagerDutyToRunbookConfig)
async def pagerduty_to_runbook(config: PagerDutyToRunbookConfig, builder: Builder):
    """Register the PagerDuty to Runbook function."""

    async def get_complete_runbook(pagerduty_url: str) -> str:
        """
        Extract alert name from PagerDuty incident and return complete matching PDF runbook.
        
        Args:
            pagerduty_url (str): PagerDuty incident URL
            
        Returns:
            str: Complete PDF runbook content with metadata
        """
        try:
            logger.info(f"Processing PagerDuty URL: {pagerduty_url}")
            
            # Step 1: Extract incident ID from URL
            match = re.search(r'/incidents?/([a-zA-Z0-9]+)', pagerduty_url, re.IGNORECASE)
            if not match:
                return "❌ Error: Invalid PagerDuty incident URL format"

            incident_id = match.group(1)
            logger.info(f"Extracted incident ID: {incident_id}")
            
            # Step 2: Get PagerDuty API token
            api_token = config.pagerduty_api_token or os.getenv('PAGERDUTY_API_TOKEN')
            if not api_token:
                return "❌ Error: PagerDuty API token is required"

            # Step 3: Fetch incident details from PagerDuty
            headers = {
                "Authorization": f"Token token={api_token}",
                "Accept": f"application/vnd.pagerduty+json;version={config.pagerduty_api_version}"
            }

            base_url = os.getenv('PAGERDUTY_API_URL', 'https://api.pagerduty.com')
            incident_url = f"{base_url}/incidents/{incident_id}"
            
            logger.info(f"Fetching from PagerDuty API: {incident_url}")

            async with httpx.AsyncClient(timeout=config.pagerduty_timeout) as client:
                incident_response = await client.get(incident_url, headers=headers)
                
                if incident_response.status_code != 200:
                    return f"❌ Error: Failed to fetch incident from PagerDuty: {incident_response.text}"

                response_json = incident_response.json()
                incident = response_json.get("incident", {})
                
                if not incident:
                    return "❌ Error: No incident data found in PagerDuty response"

                # Step 4: Extract alert name from incident title
                incident_title = incident.get("title", "")
                incident_description = incident.get("description", "")
                service_name = incident.get("service", {}).get("name", "")
                
                logger.info(f"Incident title: {incident_title}")
                logger.info(f"Service name: {service_name}")
                
                # Extract key alert terms from title
                alert_name = incident_title
                
                # Step 5: Search Milvus for matching PDF
                logger.info(f"Searching Milvus for alert: {alert_name}")
                
                # Import required libraries for Milvus
                from langchain_milvus import Milvus
                from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
                
                # Set up embedder
                embedder = NVIDIAEmbeddings(
                    model=config.embedding_model,
                    truncate="END"
                )
                
                # Set up Milvus connection
                connection_args = {
                    "uri": config.milvus_uri,
                    "user": config.milvus_user,
                    "password": config.milvus_password,
                    "db_name": config.milvus_db_name
                }
                
                vector_store = Milvus(
                    embedding_function=embedder,
                    collection_name=config.collection_name,
                    connection_args=connection_args
                )
                
                # Search for matching documents
                search_results = await vector_store.asimilarity_search(
                    alert_name,
                    k=50  # Get more results to ensure we find the right PDF
                )
                
                if not search_results:
                    return f"❌ No matching runbook found for alert: {alert_name}"
                
                # Step 6: Find best matching PDF based on filename similarity to alert name
                pdf_filenames = set()
                
                for doc in search_results:
                    filename = doc.metadata.get('filename', 'unknown')
                    pdf_filenames.add(filename)
                
                logger.info(f"Found PDFs: {list(pdf_filenames)}")
                logger.info(f"Alert name to match: {alert_name}")
                
                # Find best matching PDF by name similarity
                selected_filename = None
                best_match_score = 0
                
                for filename in pdf_filenames:
                    # Calculate similarity between alert name and PDF filename
                    # Remove common extensions and normalize
                    clean_filename = filename.replace('.pdf', '').replace('_', ' ').replace('-', ' ').lower()
                    clean_alert = alert_name.replace('_', ' ').replace('-', ' ').lower()
                    
                    # Check for exact matches first
                    if clean_alert in clean_filename or clean_filename in clean_alert:
                        match_score = 100  # Perfect match
                    else:
                        # Calculate word overlap
                        alert_words = set(clean_alert.split())
                        filename_words = set(clean_filename.split())
                        
                        if alert_words and filename_words:
                            overlap = len(alert_words.intersection(filename_words))
                            total_words = len(alert_words.union(filename_words))
                            match_score = (overlap / total_words) * 100 if total_words > 0 else 0
                        else:
                            match_score = 0
                    
                    logger.info(f"PDF: {filename} -> Match score: {match_score:.2f}%")
                    
                    if match_score > best_match_score:
                        best_match_score = match_score
                        selected_filename = filename
                
                # Only proceed if we have a good match (50% or higher similarity)
                if selected_filename and best_match_score >= 50.0:
                    logger.info(f"Selected PDF: {selected_filename} (match score: {best_match_score:.2f}%)")
                else:
                    if selected_filename:
                        logger.warning(f"Best match '{selected_filename}' has low similarity ({best_match_score:.2f}%) - below 50% threshold")
                    else:
                        logger.warning(f"No matching runbook found for alert: {alert_name}")
                    logger.info(f"Available PDFs: {list(pdf_filenames)}")
                    return f"❌ No relevant runbook found for alert: {alert_name}\n\nBest match: {selected_filename} ({best_match_score:.2f}% similarity - below 50% threshold)\n\nAvailable runbooks: {', '.join(pdf_filenames)}"
                
                # Step 7: Get ALL chunks from the selected PDF using metadata filtering
                logger.info(f"Getting all chunks from PDF: {selected_filename}")
                
                all_pdf_chunks = []
                
                try:
                    # Method 1: Use direct PyMilvus collection query (most reliable)
                    logger.info(f"Using direct collection query for filename: {selected_filename}")
                    
                    # Import pymilvus for direct collection access
                    from pymilvus import connections, Collection
                    
                    # Connect to Milvus directly
                    connections.connect(
                        alias="default",
                        uri=config.milvus_uri,
                        user=config.milvus_user,
                        password=config.milvus_password,
                        db_name=config.milvus_db_name
                    )
                    
                    # Get the collection
                    collection = Collection(config.collection_name)
                    collection.load()
                    
                    # Query with metadata filter - get ALL chunks for this filename
                    query_results = collection.query(
                        expr=f'filename == "{selected_filename}"',
                        output_fields=["text", "filename", "page_number", "chunk_index", "total_pages", "document_type"],
                        limit=16384  # Maximum limit to get all chunks
                    )
                    
                    logger.info(f"Direct collection query found {len(query_results)} chunks")
                    
                    for result in query_results:
                        all_pdf_chunks.append({
                            'chunk_index': result.get('chunk_index', 0),
                            'page_number': result.get('page_number', 0),
                            'text': result.get('text', ''),
                            'metadata': {
                                'filename': result.get('filename', ''),
                                'page_number': result.get('page_number', 0),
                                'chunk_index': result.get('chunk_index', 0),
                                'total_pages': result.get('total_pages', 0),
                                'document_type': result.get('document_type', '')
                            }
                        })
                        
                except Exception as e:
                    logger.warning(f"Direct collection query failed: {e}")
                    
                    # Fallback: Use chunks from initial search results for this filename
                    logger.info("Using chunks from initial search as fallback")
                    for doc in search_results:
                        if doc.metadata.get('filename') == selected_filename:
                            all_pdf_chunks.append({
                                'chunk_index': doc.metadata.get('chunk_index', 0),
                                'page_number': doc.metadata.get('page_number', 0),
                                'text': doc.page_content,
                                'metadata': doc.metadata
                            })
                
                # Sort chunks by chunk_index to maintain document order
                all_pdf_chunks.sort(key=lambda x: x['chunk_index'])
                
                logger.info(f"Total unique chunks found for {selected_filename}: {len(all_pdf_chunks)}")
                
                if not all_pdf_chunks:
                    return f"❌ No chunks found for PDF: {selected_filename}"
                
                # Debug: Log the actual number of chunks and their size
                total_text_length = sum(len(chunk['text']) for chunk in all_pdf_chunks)
                logger.info(f"DEBUGGING: Total chunks: {len(all_pdf_chunks)}, Total text length: {total_text_length} characters")
                
                # Log chunk indices to verify completeness
                chunk_indices = [chunk['chunk_index'] for chunk in all_pdf_chunks]
                logger.info(f"Chunk indices found: {sorted(chunk_indices)}")
                
                # Check for gaps in chunk sequence
                missing_chunks = set()
                if chunk_indices:
                    min_chunk = min(chunk_indices)
                    max_chunk = max(chunk_indices)
                    expected_chunks = set(range(min_chunk, max_chunk + 1))
                    found_chunks = set(chunk_indices)
                    missing_chunks = expected_chunks - found_chunks
                    
                    if missing_chunks:
                        logger.warning(f"Missing chunks detected: {sorted(missing_chunks)}")
                        logger.info(f"Expected {len(expected_chunks)} chunks, found {len(found_chunks)} chunks")
                    else:
                        logger.info(f"Complete sequence: {len(found_chunks)} chunks from {min_chunk} to {max_chunk}")
                
                # Step 8: Format complete output
                total_chunks = len(all_pdf_chunks)
                total_pages = max([chunk['page_number'] for chunk in all_pdf_chunks], default=1)
                
                output = f"PagerDuty Alert: {alert_name}\n"
                output += f"Matching Runbook: {selected_filename}\n"
                output += f"Total Chunks Retrieved: {total_chunks}\n"
                output += f"Total Pages: {total_pages}\n"
                output += f"Service: {service_name}\n"
                output += f"Incident ID: {incident_id}\n"
                output += f"Chunk Range: {min(chunk_indices) if chunk_indices else 0} to {max(chunk_indices) if chunk_indices else 0}\n"
                if missing_chunks:
                    output += f"⚠️ Missing Chunks: {sorted(missing_chunks)}\n"
                output += "\n"
                output += "="*80 + "\n"
                output += "COMPLETE RUNBOOK CONTENT\n"
                output += "="*80 + "\n\n"
                
                # Output all chunks in order
                for chunk in all_pdf_chunks:
                    chunk_num = chunk['chunk_index']
                    page_num = chunk['page_number']
                    text = chunk['text']
                    
                    output += f"[Chunk {chunk_num} - Page {page_num}]\n"
                    output += f"{text}\n\n"
                
                output += "="*80 + "\n"
                output += "END OF RUNBOOK\n"
                output += "="*80 + "\n"
                
                logger.info(f"Successfully compiled runbook with {total_chunks} chunks")
                return output
                
        except Exception as e:
            logger.exception(f"Error in pagerduty_to_runbook: {e}")
            return f"❌ Error processing request: {str(e)}"

    yield FunctionInfo.from_fn(
        get_complete_runbook,
        description="Extract alert name from PagerDuty incident and return complete matching PDF runbook content"
    ) 