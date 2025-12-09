# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import json
from collections.abc import AsyncGenerator
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from nat.builder.builder import Builder
from nat.builder.function import FunctionGroup
from nat.cli.register_workflow import register_function_group
from nat.data_models.function import FunctionGroupBaseConfig


class RetailToolsConfig(FunctionGroupBaseConfig, name="retail_tools"):
    """Configuration for the retail agent tools."""

    data_dir: Path = Field(
        default_factory=lambda: Path(__file__).parent / "data",
        description="Directory containing the customer and product JSON files.",
    )
    include: list[str] = Field(
        default_factory=lambda: [
            "get_customer_info",
            "get_product_info",
            "get_all_products",
            "write_review",
            "send_email",
            "update_customer_info",
        ],
        description="The list of functions to include in the retail tools function group.",
    )


class WriteReviewParams(BaseModel):
    """Parameters for writing a product review."""

    customer_email: str = Field(..., description="The customer's email address")
    product_name: str = Field(..., description="The name or ID of the product")
    rating: int = Field(..., ge=1, le=5, description="Rating between 1 and 5")
    review_text: str = Field(..., description="The review text content")


class SendEmailParams(BaseModel):
    """Parameters for sending an email."""

    recipient_email: str = Field(..., description="The recipient's email address")
    content: str = Field(..., description="The email content")
    cc: str | None = Field(None, description="Optional CC email address")


class UpdateCustomerInfoParams(BaseModel):
    """Parameters for updating customer information with a new order."""

    customer_email: str = Field(..., description="The customer's email address")
    product_name: str = Field(..., description="The name or ID of the product")
    quantity: int = Field(..., gt=0, description="The quantity to order (must be positive)")


@register_function_group(config_type=RetailToolsConfig)
async def retail_tools(_config: RetailToolsConfig, _builder: Builder) -> AsyncGenerator[FunctionGroup, None]:
    """Create and register the retail agent function group.

    Args:
        _config: Retail tools function group configuration.
        _builder: Workflow builder (unused).

    Yields:
        FunctionGroup: The configured retail tools function group with customer and product management functions.
    """
    # Load data files
    customers_file = _config.data_dir / "customers.json"
    products_file = _config.data_dir / "products.json"

    try:
        with open(customers_file) as f:
            customers_data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        raise RuntimeError("Failed to load customers data") from e

    try:
        with open(products_file) as f:
            products_data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        raise RuntimeError("Failed to load products data") from e

    group = FunctionGroup(config=_config)

    async def _get_customer_info(email: str) -> dict[str, Any]:
        """Look up customer information by email address.

        Args:
            email: The customer's email address.

        Returns:
            Customer information including id, name, past orders, total orders, total spent, and past reviews.
        """
        for customer in customers_data:
            if customer["email"].lower() == email.lower():
                return customer

        return {
            "error": f"No customer found with email: {email}",
            "message": "This appears to be a new customer. They have no purchase history.",
        }

    async def _get_product_info(product_identifier: str) -> dict[str, Any]:
        """Retrieve detailed information about a specific product.

        Args:
            product_identifier: Either the product ID (for example PROD001) or product name.

        Returns:
            Product information including id, name, description, price, stock, and reviews.
        """
        for product in products_data:
            if (
                product["id"].lower() == product_identifier.lower()
                or product["name"].lower() == product_identifier.lower()
            ):
                return product

        return {"error": f"No product found with identifier: {product_identifier}"}

    async def _get_all_products(dummy: str = "") -> list[dict[str, Any]]:
        """Retrieve a list of all available products.

        Returns:
            List of all products with their basic information (id, name, description, price, stock).
        """
        del dummy
        return [
            {
                "id": p["id"],
                "name": p["name"],
                "description": p["description"],
                "price": p["price"],
                "stock": p["stock"],
                "average_rating": (
                    sum(r["rating"] for r in p["reviews"]) / len(p["reviews"]) if p["reviews"] else "No ratings yet"
                ),
                "review_count": len(p["reviews"]),
            }
            for p in products_data
        ]

    async def _write_review(params: WriteReviewParams) -> dict[str, Any]:
        """Submit a product review (mock function - does not persist data).

        Args:
            params: WriteReviewParams with customer_email, product_name, rating, and review_text.

        Returns:
            Success confirmation with review details.
        """
        # Check if customer exists
        customer = await _get_customer_info(params.customer_email)
        if "error" in customer:
            return {
                "error": "Customer not found",
                "message": "Only existing customers can write reviews.",
            }

        # Check if product exists
        product = await _get_product_info(params.product_name)
        if "error" in product:
            return product

        # Mock success response
        return {
            "success": True,
            "message": f"Review submitted successfully for {product['name']}",
            "review": {
                "customer_name": customer["name"],
                "product_name": product["name"],
                "rating": params.rating,
                "review_text": params.review_text,
            },
            "note": "This is a mock operation - the review was not actually saved to the database.",
        }

    async def _send_email(params: SendEmailParams) -> dict[str, Any]:
        """Send an email to a customer (mock function - no actual email sent).

        Args:
            params: SendEmailParams with recipient_email, content, and optional cc.

        Returns:
            Success confirmation with email details.
        """
        return {
            "success": True,
            "message": "Email sent successfully",
            "email_details": {
                "to": params.recipient_email,
                "cc": params.cc or "None",
                "content": params.content,
                "timestamp": "2024-11-25T10:00:00Z",
            },
            "note": "This is a mock operation - no actual email was sent.",
        }

    async def _update_customer_info(params: UpdateCustomerInfoParams) -> dict[str, Any]:
        """Update customer information with a new order (mock function - does not persist data).

        Args:
            params: UpdateCustomerInfoParams with customer_email, product_name, and quantity.

        Returns:
            Success confirmation with updated order details.
        """
        # Check if customer exists
        customer = await _get_customer_info(params.customer_email)
        if "error" in customer:
            return {
                "error": "Customer not found",
                "message": "Cannot update information for non-existent customer.",
            }

        # Check if product exists
        product = await _get_product_info(params.product_name)
        if "error" in product:
            return product

        # Check stock availability
        if product["stock"] < params.quantity:
            return {
                "error": "Insufficient stock",
                "message": f"Only {product['stock']} units of {product['name']} are available.",
            }

        # Calculate order total
        order_total = product["price"] * params.quantity

        # Mock success response
        return {
            "success": True,
            "message": f"Order placed successfully for {customer['name']}",
            "order_details": {
                "customer_name": customer["name"],
                "customer_email": customer["email"],
                "product_name": product["name"],
                "product_id": product["id"],
                "quantity": params.quantity,
                "unit_price": product["price"],
                "total": order_total,
                "new_total_orders": customer["total_orders"] + 1,
                "new_total_spent": customer["total_spent"] + order_total,
            },
            "note": "This is a mock operation - the order was not actually saved to the database.",
        }

    # Add functions to the group
    group.add_function(name="get_customer_info", fn=_get_customer_info, description=_get_customer_info.__doc__)
    group.add_function(name="get_product_info", fn=_get_product_info, description=_get_product_info.__doc__)
    group.add_function(name="get_all_products", fn=_get_all_products, description=_get_all_products.__doc__)
    group.add_function(name="write_review", fn=_write_review, description=_write_review.__doc__)
    group.add_function(name="send_email", fn=_send_email, description=_send_email.__doc__)
    group.add_function(
        name="update_customer_info",
        fn=_update_customer_info,
        description=_update_customer_info.__doc__,
    )

    yield group
