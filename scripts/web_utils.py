# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import asyncio
import logging
import os
from collections.abc import Awaitable
from urllib.parse import urljoin
from urllib.parse import urlparse

import httpx

logger = logging.getLogger(__name__)
ScrapeResult = dict[str, str | None]
_REDIRECT_STATUS_CODES = {301, 302, 303, 307, 308}
_MAX_REDIRECTS = 5


async def _wrap_request(f: Awaitable[httpx.Response], url: str) -> ScrapeResult:
    try:
        resp = {"url": url, "content": (await f).text}
    except Exception as e:
        logger.exception("Error in _wrap_request for %s: %s", url, e)
        resp = {"url": url, "content": None, "exception": f"{e}"}
    return resp


def _url_origin(url: str) -> tuple[str, str | None, int | None]:
    parsed_url = urlparse(url)
    hostname = parsed_url.hostname.lower() if parsed_url.hostname else None
    port = parsed_url.port
    if port is None:
        if parsed_url.scheme == "https":
            port = 443
        elif parsed_url.scheme == "http":
            port = 80

    return parsed_url.scheme.lower(), hostname, port


async def _get_with_same_origin_redirects(client: httpx.AsyncClient,
                                          url: str,
                                          headers: dict[str, str],
                                          *,
                                          max_redirects: int = _MAX_REDIRECTS) -> httpx.Response:
    current_url = url
    original_origin = _url_origin(url)

    for _ in range(max_redirects + 1):
        response = await client.get(current_url, headers=headers, follow_redirects=False)
        if response.status_code not in _REDIRECT_STATUS_CODES:
            return response

        redirect_location = response.headers.get("location")
        if not redirect_location:
            raise ValueError(f"Redirect response for {current_url} did not include a Location header")

        redirect_url = urljoin(current_url, redirect_location)
        if _url_origin(redirect_url) != original_origin:
            raise ValueError(f"Redirect target {redirect_url} is not same-origin with {url}")

        current_url = redirect_url

    raise ValueError(f"Exceeded {max_redirects} redirects while scraping {url}")


async def scrape(urls: list[str] | str,
                 headers: dict[str, str] | None = None) -> tuple[list[ScrapeResult], list[ScrapeResult]]:
    """
    Retrieve the page content for a given list of urls.

    Args:
      urls (list): List of urls (or a single url string)
      headers (dict): Dictionary of headers to use in the request

    Returns (Tuple(list[dict], list[dict])): Tuple containing lists of dictionaries:
      "responses" which contains the urls and content of each successful request
      "failures" which contains the urls and exceptions for each unsuccessful request
    """
    headers = {'user-agent': 'Mozilla/5.0'} if not headers else headers
    urls = [urls] if isinstance(urls, str) else urls
    responses = []
    failures = []
    async with httpx.AsyncClient() as client:
        tasks = [_wrap_request(_get_with_same_origin_redirects(client, url, headers), url) for url in urls]
        for response_future in asyncio.as_completed(tasks):
            response = await response_future
            if response.get("content"):
                responses.append(response)
            else:
                failures.append(response)
    logger.debug(responses)
    return responses, failures


def get_file_path_from_url(url: str, base_path: str) -> tuple[str, str]:
    """
    Generate a filepath based on the url, using the domain as the parent directory.

    Resulting filepaths take the form {base_path}/{domain}/{page_name}
    Examples:
       http://mydomain.com/articles/generative_ai -> {base_path}/mydomain/articles_generative_ai

    Args:
     url (str): The url from which to generate a file name
     base_path (str): The base path to build the new path from

    Returns:
     filepath (str): File path based generated from the URL
     directory (str): Path to the parent directory
    """
    short_url, domain = _get_short_url(url)
    short_url = short_url.replace("/", "_")
    domain = domain.replace("/", "_")
    directory = os.path.join(base_path, domain)
    file_path = os.path.join(base_path, domain, short_url)
    return file_path, directory


def cache_html(input_dict: ScrapeResult, base_path: str = ".") -> tuple[ScrapeResult, str | None]:
    """
    Save HTML data to disk.

    Args:
     input_dict (dict): Dictionary of HTML content containing the url and content
     base_path (str): Base path under which all directories and files will be created

    Returns
     input_dict (dict): Original input
     file_path (str): Path to the saved data
    """
    url = input_dict.get("url")
    data = input_dict.get("content")
    if not url or not data:
        logger.error("Invalid input for saving to cache for: %s", input_dict)
        return input_dict, None
    file_path, directory = get_file_path_from_url(url, base_path)

    os.makedirs(directory, exist_ok=True)
    try:
        with open(file_path, 'w', encoding="utf-8") as f:
            f.write(data)
    except Exception:
        logger.error("Unable to save data for %s", url, exc_info=True)
        raise
    return input_dict, file_path


def _get_short_url(url: str) -> tuple[str, str]:
    path = url.rsplit("://", maxsplit=1)[-1].split("www.")[-1]
    path_components = path.split("/")
    domain = path_components[0]
    short_url = "/".join(path_components[1:])
    return short_url, domain
