import logging

import json
from urllib.parse import quote

from pydantic import Field

from nat.builder.builder import Builder
from nat.builder.function_info import FunctionInfo
from nat.cli.register_workflow import register_function
from nat.data_models.function import FunctionBaseConfig


logger = logging.getLogger(__name__)


class LocalRagFunctionConfig(FunctionBaseConfig, name="local_rag"):
    """
    NAT function template. Please update the description.
    """
    
    base_url: str = Field(description="Local / Custom RAG URL")
    #prompt: str = Field(default="Hello", description="The prompt")
    max_records: int = Field(default="1", description="Maximum number of records to be retrieved")


@register_function(config_type=LocalRagFunctionConfig)
async def local_rag_function(
    config: LocalRagFunctionConfig, builder: Builder
):
    import httpx 
    async with httpx.AsyncClient(verify=False, headers={
                "accept": "application/json", "Content-Type": "application/json"
        }) as client:

            async def _response_fn(query: str) -> str:
                """
                This tool retrieve relevant context for the given question
                """
                logger.info("Your query is %s", query)

                # configure params for RAG endpoint and doc search
                url = f"{config.base_url}"
                # payload = {"prompt": quote(query, safe=""), "max_records": config.max_records}
                payload = {"prompt": query, "max_records": config.max_records}

                # send configured payload to running chain server
                logger.debug("Sending request to the RAG endpoint %s", url)

                
                url_encoded_prompt = quote(query, safe="")
                request =  f"{url}?prompt={url_encoded_prompt}&max_records={config.max_records}"

                logger.info("Your URL is %s", request)

                # response = await client.get(url, params=payload)
                response = await client.get(request)

                response.raise_for_status()
                results = response.json()

                logger.info("The results are %s", results)

                if len(results["records"]) == 0:
                    return ""

                # parse docs from LangChain/LangGraph Document object to string
                parsed_docs = []

                # iterate over results and store parsed content

                num_records = results["num_records"]
                records = results["records"]
                for i in range(num_records):
                    link = records[i]["_links"]["self"]["href"]
                    content = records[i]["chunk"]
                    parsed_document = f'<Document"/> link={link}\n"{content}\n</Document>'
                    parsed_docs.append(parsed_document)

                # combine parsed documents into a single string
                internal_search_docs = "\n\n---\n\n".join(parsed_docs)
                return internal_search_docs

            yield FunctionInfo.from_fn(
                _response_fn,
                description=("This tool retrieves relevant documents for a given user query."
                            "This will return relevant documents from the selected collection."))