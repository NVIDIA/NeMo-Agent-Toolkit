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

import datetime

import pytest

from nat_adk_demo.location import CityLocation
from nat_adk_demo.location import geocode_city
from nat_adk_demo.nat_time_tool import TimeMCPToolConfig
from nat_adk_demo.nat_time_tool import get_time_for_city
from nat_adk_demo.weather_update_tool import WeatherToolConfig
from nat_adk_demo.weather_update_tool import get_current_weather


@pytest.mark.parametrize(
    ("city", "expected_timezone"),
    [
        ("London", "Europe/London"),
        ("Tokyo, Japan", "Asia/Tokyo"),
    ],
)
async def test_geocode_city_resolves_locations(monkeypatch, city: str, expected_timezone: str):

    def mock_fetch_json(url, params, timeout_seconds):
        assert url == "https://example.test/geocode"
        assert timeout_seconds == 1
        city_name = str(params["name"]).split(",", maxsplit=1)[0]
        return {
            "results": [{
                "name": city_name,
                "country": "United Kingdom" if city_name == "London" else "Japan",
                "latitude": 51.5072 if city_name == "London" else 35.6895,
                "longitude": -0.1276 if city_name == "London" else 139.6917,
                "timezone": expected_timezone,
            }]
        }

    monkeypatch.setattr("nat_adk_demo.location.fetch_json", mock_fetch_json)

    location = await geocode_city(city, "https://example.test/geocode", 1)

    assert location is not None
    assert location.name in city
    assert location.timezone == expected_timezone


async def test_get_current_weather_uses_resolved_city(monkeypatch):
    london = CityLocation(name="London",
                          country="United Kingdom",
                          latitude=51.5072,
                          longitude=-0.1276,
                          timezone="Europe/London")

    async def mock_geocode_city(city, geocoding_url, timeout_seconds):
        assert city == "London"
        assert geocoding_url == "https://example.test/geocode"
        assert timeout_seconds == 1
        return london

    async def mock_fetch_current_weather(location, config):
        assert location == london
        assert config.forecast_url == "https://example.test/forecast"
        return {
            "current": {
                "temperature_2m": 12.3,
                "apparent_temperature": 10.8,
                "relative_humidity_2m": 78,
                "precipitation": 0,
                "weather_code": 3,
                "wind_speed_10m": 9.4,
            }
        }

    monkeypatch.setattr("nat_adk_demo.weather_update_tool.geocode_city", mock_geocode_city)
    monkeypatch.setattr("nat_adk_demo.weather_update_tool._fetch_current_weather", mock_fetch_current_weather)

    result = await get_current_weather(
        "London",
        WeatherToolConfig(geocoding_url="https://example.test/geocode",
                          forecast_url="https://example.test/forecast",
                          timeout_seconds=1),
    )

    assert "London, United Kingdom" in result
    assert "overcast" in result
    assert "12.3 degrees Celsius" in result


async def test_geocode_city_retries_comma_qualified_query(monkeypatch):
    seen_queries = []

    def mock_fetch_json(url, params, timeout_seconds):
        seen_queries.append(params["name"])
        if params["name"] == "Tokyo, Japan":
            return {}
        return {
            "results": [{
                "name": "Tokyo",
                "country": "Japan",
                "latitude": 35.6895,
                "longitude": 139.6917,
                "timezone": "Asia/Tokyo",
            }]
        }

    monkeypatch.setattr("nat_adk_demo.location.fetch_json", mock_fetch_json)

    location = await geocode_city("Tokyo, Japan", "https://example.test/geocode", 1)

    assert seen_queries == ["Tokyo, Japan", "Tokyo"]
    assert location is not None
    assert location.display_name == "Tokyo, Japan"


async def test_get_time_for_city_uses_resolved_timezone(monkeypatch):
    tokyo = CityLocation(name="Tokyo", country="Japan", latitude=35.6895, longitude=139.6917, timezone="Asia/Tokyo")

    async def mock_geocode_city(city, geocoding_url, timeout_seconds):
        assert city == "Tokyo, Japan"
        assert geocoding_url == "https://example.test/geocode"
        assert timeout_seconds == 1
        return tokyo

    class MockDateTime(datetime.datetime):

        @classmethod
        def now(cls, tz=None):
            return cls(2026, 5, 14, 9, 30, 0, tzinfo=tz)

    monkeypatch.setattr("nat_adk_demo.nat_time_tool.geocode_city", mock_geocode_city)
    monkeypatch.setattr("nat_adk_demo.nat_time_tool.datetime.datetime", MockDateTime)

    result = await get_time_for_city(
        "Tokyo, Japan",
        TimeMCPToolConfig(geocoding_url="https://example.test/geocode", timeout_seconds=1),
    )

    assert result == "The current time in Tokyo, Japan is 2026-05-14 09:30:00 JST+0900."
