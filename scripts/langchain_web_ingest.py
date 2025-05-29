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

import logging
import os
from uuid import uuid4

from langchain_community.document_loaders import BSHTMLLoader
from langchain_milvus import Milvus
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from web_utils import cache_html
from web_utils import get_file_path_from_url
from web_utils import scrape

logger = logging.getLogger(__name__)


async def main(*,
               urls: list[str],
               milvus_uri: str,
               collection_name: str,
               clean_cache: bool = True,
               embedding_model: str = "nvidia/nv-embedqa-e5-v5",
               base_path: str = "./.tmp/data"):

    embedder = NVIDIAEmbeddings(model=embedding_model, truncate="END")
    vector_store = Milvus(
        embedding_function=embedder,
        collection_name=collection_name,
        connection_args={"uri": milvus_uri},
    )

    filenames = [
        get_file_path_from_url(url, base_path)[0] for url in urls
        if os.path.exists(get_file_path_from_url(url, base_path)[0])
    ]
    urls_to_scrape = [url for url in urls if get_file_path_from_url(url, base_path)[0] not in filenames]
    if filenames:
        logger.info("Loading %s from cache", filenames)
    if len(urls_to_scrape) > 0:
        logger.info("Scraping: %s", urls_to_scrape)
        html_data, err = await scrape(urls)
        if err:
            logger.info("Failed to scrape %s", {[f['url'] for f in err]})
        filenames.extend([cache_html(data, base_path)[1] for data in html_data if html_data])

    doc_ids = []
    for filename in filenames:

        logger.info("Parsing %s into documents", filename)
        loader = BSHTMLLoader(filename)
        splitter = RecursiveCharacterTextSplitter()
        docs = loader.load()
        docs = splitter.split_documents(docs)

        if not isinstance(docs, list):
            docs = [docs]

        ids = [str(uuid4()) for _ in range(len(docs))]
        logger.info("Adding %s document chunks to Milvus collection %s", len(docs), collection_name)
        doc_ids.extend(await vector_store.aadd_documents(documents=docs, ids=ids))
        logger.info("Ingested %s document chunks", len(doc_ids))
        if clean_cache:
            logger.info("Removing %s", filename)
            os.remove(filename)

    return doc_ids


if __name__ == "__main__":
    import argparse
    import asyncio

    CUDA_URLS = [
        "https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html",
        "https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html",
        "https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html",
        "https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html",
    ]
    CUDA_COLLECTION_NAME = "cuda_docs"
    DEFAULT_URI = "http://localhost:19530"

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--urls", default=CUDA_URLS, action="append", help="Urls to scrape for RAG context")
    parser.add_argument("--collection_name", "-n", default=CUDA_COLLECTION_NAME, help="Collection name for the data.")
    parser.add_argument("--milvus_uri", "-u", default=DEFAULT_URI, help="Milvus host URI")
    parser.add_argument("--clean_cache", default=False, help="If true, deletes local files", action="store_true")
    args = parser.parse_args()

    asyncio.run(
        main(
            urls=args.urls,
            milvus_uri=args.milvus_uri,
            collection_name=args.collection_name,
            clean_cache=args.clean_cache,
        ))
