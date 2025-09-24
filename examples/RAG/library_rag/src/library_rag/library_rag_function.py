import logging

from pydantic import Field

from nat.builder.builder import Builder
from nat.builder.function_info import FunctionInfo
from nat.cli.register_workflow import register_function
from nat.data_models.function import FunctionBaseConfig

logger = logging.getLogger(__name__)


class LibraryRagFunctionConfig(FunctionBaseConfig, name="library_rag"):
    """
    NAT function template. Please update the description.
    """
    base_url: str = Field(description="Local / Custom RAG URL")
    #prompt: str = Field(default="Hello", description="The prompt")
    reranker_top_k: int = Field(default=2, description="Maximum number of records to be retrieved") #TODO: Modify the descriptions
    vdb_top_k: int = Field(default=10, description="Maximum number of records to be retrieved")
    vdb_endpoint: str = Field(default="", description="Maximum number of records to be retrieved")
    collection_names: list[str] = Field(default="1", description="Maximum number of records to be retrieved")
    enable_query_rewriting: bool = Field(default=True, description="Maximum number of records to be retrieved")
    enable_reranker: bool = Field(default=True, description="Maximum number of records to be retrieved")

@register_function(config_type=LibraryRagFunctionConfig)
def library_rag_function(
    config: LibraryRagFunctionConfig, builder: Builder
):
    import aiohttp
    # Implement your function logic here
    async def _response_fn(query: str) -> str:
        url = f"{config.base_url}/v1/search"
        payload={
        "query": f"{query}",
        "reranker_top_k": f"{config.reranker_top_k}",
        "vdb_top_k": f"{config.vdb_top_k}",
        "vdb_endpoint": f"{config.vdb_endpoint}",
        "collection_names": f"{config.collection_names}", # Multiple collection retrieval can be used by passing multiple collection names
        "enable_query_rewriting": f"{config.enable_query_rewriting}",
        "enable_reranker": f"{config.enable_reranker}",}

        logger.info("Your query is %s", query)


        async with aiohttp.ClientSession() as session:
            try:
                logger.debug("Sending request to the RAG endpoint %s", url)

                #async with session.post(url=url, json=payload) as response:
                results = await session.post(url=url, json=payload).json()

                logger.info("The results are %s", results)

                if results["total_results"] == 0:
                    yield ""

                # parse docs from LangChain/LangGraph Document object to string
                parsed_docs = []

                # iterate over results and store parsed content

                num_records = results["total_results"]
                records = results["results"]
                for i in range(num_records):
                    document_id = records[i]["document_id"]
                    content = records[i]["content"]
                    parsed_document = f'<Document"/> document_id={document_id}\n"{content}\n</Document>'
                    parsed_docs.append(parsed_document)

                # combine parsed documents into a single string
                internal_search_docs = "\n\n---\n\n".join(parsed_docs)
                yield internal_search_docs


            except aiohttp.ClientError as e:
                print(f"Error: {e}")

        yield FunctionInfo.from_fn(
                _response_fn,
                description=("This tool retrieves relevant documents for a given user query."
                            "This will return relevant documents from the selected collection."))
