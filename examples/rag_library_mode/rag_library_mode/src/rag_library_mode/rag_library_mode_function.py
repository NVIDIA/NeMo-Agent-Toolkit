import logging

from pydantic import Field

from nat.builder.builder import Builder
from nat.builder.function_info import FunctionInfo
from nat.cli.register_workflow import register_function
from nat.data_models.function import FunctionBaseConfig

from nvidia_rag import NvidiaRAG, NvidiaRAGIngestor

import json
import base64
from IPython.display import display, Image, Markdown


logger = logging.getLogger(__name__)


class RagLibraryModeFunctionConfig(FunctionBaseConfig, name="rag_library_mode"):
    """
    This tool retrieves relevant documents for a given user query. The input query is mapped to the most appropriate
    Milvus collection database. This will return relevant documents from the selected collection.
    """
    base_url: str = Field(description="The base url used to connect to the milvus database.")
    reranker_top_k: int = Field(default=100, description="The number of results to return from the milvus database.")
    vdb_top_k: int = Field(default=10, description="The number of results to return from the milvus database.")
    collection_names: list = Field(default=["cuda_docs"],
                                   description="The list of available collection names.")
    


@register_function(config_type=RagLibraryModeFunctionConfig)
async def rag_library_mode_function(
    config: RagLibraryModeFunctionConfig, builder: Builder
):

    def parse_search_citations(citations):

        parsed_docs = []
        
        for idx, citation in enumerate(citations.results):
        # If using pydantic models, citation fields may be attributes, not dict keys
            content = getattr(citation, 'content', '')
            doc_name = getattr(citation, 'document_name', f'Citation {idx+1}')
            parsed_document = f'<Document source="{doc_name}"/>\n{content}\n</Document>'
            parsed_docs.append(parsed_document)

            # combine parsed documents into a single string
            internal_search_docs = "\n\n---\n\n".join(parsed_docs)
            return internal_search_docs

    async def _response_fn(query: str) -> str:
        # Process the input_message and generate output

        rag = NvidiaRAG()
        ingestor = NvidiaRAGIngestor()

        # Just to debug
        response = ingestor.get_documents(
        collection_name=config.collection_names,
        vdb_endpoint=config.base_url,
        )
        logger.info(f"***** {response}")
        
        return parse_search_citations(rag.search(
            query=f"{query}",
            collection_names=config.collection_names,
            reranker_top_k=config.reranker_top_k,
            vdb_top_k=config.vdb_top_k,
        ))  

    try:
        yield FunctionInfo.create(single_fn=_response_fn)
    except GeneratorExit:
        logger.warning("Function exited early!")
    finally:
        logger.info("Cleaning up rag_library_mode workflow.")