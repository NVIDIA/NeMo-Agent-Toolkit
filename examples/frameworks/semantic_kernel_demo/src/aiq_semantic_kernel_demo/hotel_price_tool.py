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

from pydantic import BaseModel

from aiq.builder.builder import Builder
from aiq.builder.function_info import FunctionInfo
from aiq.cli.register_workflow import register_function
from aiq.data_models.function import FunctionBaseConfig


class HotelPriceToolConfig(FunctionBaseConfig, name="hotel_price"):
    data_path: str = "examples/frameworks/semantic_kernel_demo/data/hotel_prices.json"
    date_format: str = "%Y-%m-%d"


class HotelOffer(BaseModel):
    name: str
    price_per_night: float
    total_price: float
    city: str
    checkin: str
    checkout: str


class HotelOffersResponse(BaseModel):
    offers: list[HotelOffer]


@register_function(config_type=HotelPriceToolConfig)
async def hotel_price(tool_config: HotelPriceToolConfig, builder: Builder):

    import json

    with open(tool_config.data_path, 'r', encoding='utf-8') as f:
        hotel_prices = json.load(f)

    search_date_format = tool_config.date_format

    async def _get_hotel_price(city: str, checkin: str, checkout: str) -> HotelOffersResponse:
        from datetime import datetime

        base_hotels = hotel_prices

        # Parse the checkin and checkout dates assuming 'YYYY-MM-DD' format
        checkin_dt = datetime.strptime(checkin, search_date_format)
        checkout_dt = datetime.strptime(checkout, search_date_format)
        nights = (checkout_dt - checkin_dt).days

        offers = []
        for hotel in base_hotels:
            total_price = hotel["price_per_night"] * nights
            offers.append(
                HotelOffer(name=hotel["name"],
                           price_per_night=hotel["price_per_night"],
                           total_price=total_price,
                           city=city,
                           checkin=checkin,
                           checkout=checkout))

        return HotelOffersResponse(offers=offers)

    yield FunctionInfo.from_fn(
        _get_hotel_price,
        description=(
            "This tool returns a list of hotels and nightly prices for the given city and checkin/checkout dates."))
