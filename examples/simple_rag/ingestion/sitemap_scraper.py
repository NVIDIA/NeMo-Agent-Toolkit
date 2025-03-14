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

import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)


def get_urls_from_sitemap(sitemaps: str | list, headers: dict = None, limit: int = None) -> list[dict]:
    """
    Get all urls listed in a sitemap or list of sitemaps. If passed a sitemap index it will recursively traveerse the
    index to get all urls across all sitemaps.

    Args:
     sitemaps: string or list of strings specifying the url for the sitemap(s)
     headers: dictionary to pass as headers to the requests session
     limit: maximum number of urls to return

    Returns:
     list of dictionaries with keys 'url', 'lastMod', 'changeFreq'. Only the url is guaranteed to be present
    """
    logger.debug("Getting urls from %s", sitemaps)
    session = requests.Session()
    headers = {'user-agent': 'Mozilla/5.0'} if not headers else headers
    session.headers.update(headers)

    sitemaps = [sitemaps] if isinstance(sitemaps, str) else sitemaps

    urls = []

    for site in sitemaps:
        urls.extend(_get_urls_from_sitemaps(site, session))
    if limit:
        return urls[:limit]
    return urls


def _get_urls_from_sitemaps(sitemap: str, session):
    logger.debug("Call to get_urls function")
    urls = []
    try:
        resp = session.get(sitemap)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.content, features="xml")
        sitemaps = soup.find_all('sitemap')
        links = soup.find_all('url')

        for elem in links:
            res = {
                "url": str(elem.loc.string) if elem.loc else None,
                "lastMod": elem.lastmod.string if elem.lastmod else None,
                "changeFreq": elem.changefreq.string if elem.changefreq else None,
            } if elem.loc.string else None
            if res:
                urls.append(res)
        for site in sitemaps:
            urls.extend(_get_urls_from_sitemaps(site.loc.string, session))
        return urls
    except Exception as e:
        logger.exception("Error pulling sitemap from  %s: %s", sitemap, e, exc_info=True)
        return urls
