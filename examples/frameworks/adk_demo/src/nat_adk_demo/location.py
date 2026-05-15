# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Shared location lookup helpers for the ADK demo tools."""

import asyncio
import json
import logging
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlencode
from urllib.parse import urlparse
from urllib.request import Request
from urllib.request import urlopen

DEFAULT_GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
DEFAULT_TIMEOUT_SECONDS = 10.0

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CityLocation:
    """Resolved city metadata returned by the geocoding service."""

    name: str
    country: str
    latitude: float
    longitude: float
    timezone: str
    admin1: str | None = None

    @property
    def display_name(self) -> str:
        parts = [self.name]
        if self.admin1 and self.admin1.casefold() != self.name.casefold():
            parts.append(self.admin1)
        if self.country:
            parts.append(self.country)
        return ", ".join(parts)


def fetch_json(url: str, params: dict[str, str | int | float], timeout_seconds: float) -> dict[str, Any]:
    """Fetch a JSON object using only the Python standard library."""

    request_url = f"{url}?{urlencode(params)}"
    parsed_url = urlparse(request_url)
    if parsed_url.scheme.lower() != "https":
        raise ValueError(f"Only HTTPS URLs are supported: {url}")

    request = Request(request_url, headers={"User-Agent": "nat-adk-demo/1.0"})

    with urlopen(request, timeout=timeout_seconds) as response:
        payload = json.loads(response.read().decode("utf-8"))

    if not isinstance(payload, dict):
        raise ValueError("Expected a JSON object response.")

    return payload


def _location_from_result(result: dict[str, Any]) -> CityLocation | None:
    try:
        name = result["name"]
        latitude = float(result["latitude"])
        longitude = float(result["longitude"])
    except (KeyError, TypeError, ValueError):
        return None

    timezone = result.get("timezone")
    if not isinstance(timezone, str) or not timezone:
        return None

    country = result.get("country")
    admin1 = result.get("admin1")

    return CityLocation(name=name,
                        country=country if isinstance(country, str) else "",
                        latitude=latitude,
                        longitude=longitude,
                        timezone=timezone,
                        admin1=admin1 if isinstance(admin1, str) else None)


async def geocode_city(city: str,
                       geocoding_url: str = DEFAULT_GEOCODING_URL,
                       timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS) -> CityLocation | None:
    """Resolve a city name to coordinates and an IANA timezone."""

    query = city.strip()
    if not query:
        return None

    queries = [query]
    if "," in query:
        queries.append(query.split(",", maxsplit=1)[0].strip())

    for candidate_query in queries:
        if not candidate_query:
            continue

        try:
            payload = await asyncio.to_thread(fetch_json,
                                              geocoding_url, {
                                                  "name": candidate_query,
                                                  "count": 1,
                                                  "language": "en",
                                                  "format": "json",
                                              },
                                              timeout_seconds)
        except Exception as ex:
            logger.warning("Failed to geocode city query %r: %s", candidate_query, ex)
            continue

        results = payload.get("results")
        if not isinstance(results, list):
            continue

        for result in results:
            if isinstance(result, dict):
                location = _location_from_result(result)
                if location is not None:
                    return location

    return None
