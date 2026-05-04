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

import json
from collections.abc import AsyncGenerator
from pathlib import Path

from pydantic import BaseModel
from pydantic import Field

from nat.builder.builder import Builder
from nat.builder.function import FunctionGroup
from nat.cli.register_workflow import register_function_group
from nat.data_models.function import FunctionGroupBaseConfig

# ============================================================================
# Data Models for Customer Data
# ============================================================================


class PastOrder(BaseModel):
    """A past order in a customer's order history."""

    product_id: str = Field(..., description="The product ID")
    product_name: str = Field(..., description="The name of the product")
    quantity: int = Field(..., description="The quantity ordered")
    date: str = Field(..., description="The date of the order")
    total: float = Field(..., description="The total cost of the order")


class Customer(BaseModel):
    """Customer information including order history."""

    id: str = Field(..., description="The unique customer ID")
    email: str = Field(..., description="The customer's email address")
    name: str = Field(..., description="The customer's full name")
    past_orders: list[PastOrder] = Field(default_factory=list, description="List of past orders")
    total_orders: int = Field(..., description="Total number of orders placed")
    total_spent: float = Field(..., description="Total amount spent by the customer")


# ============================================================================
# Data Models for Product Data
# ============================================================================


class ProductReview(BaseModel):
    """A review for a product."""

    customer_id: str = Field(..., description="The ID of the customer who wrote the review")
    customer_name: str = Field(..., description="The name of the customer who wrote the review")
    rating: int = Field(..., ge=1, le=5, description="The rating given (1-5)")
    review: str = Field(..., description="The review text")


class Product(BaseModel):
    """Full product information including reviews."""

    id: str = Field(..., description="The unique product ID")
    name: str = Field(..., description="The product name")
    description: str = Field(..., description="The product description")
    price: float = Field(..., description="The product price")
    stock: int = Field(..., description="The current stock level")
    reviews: list[ProductReview] = Field(default_factory=list, description="List of product reviews")


class ProductSummary(BaseModel):
    """Summarized product information returned by get_all_products."""

    id: str = Field(..., description="The unique product ID")
    name: str = Field(..., description="The product name")
    description: str = Field(..., description="The product description")
    price: float = Field(..., description="The product price")
    stock: int = Field(..., description="The current stock level")
    average_rating: float | str = Field(..., description="The average rating or 'No ratings yet'")
    review_count: int = Field(..., description="The number of reviews")
    review_texts: list[str] = Field(default_factory=list, description="List of review texts")


# ============================================================================
# Response Models for Actions
# ============================================================================


class ReviewDetails(BaseModel):
    """Details of a submitted review."""

    customer_name: str = Field(..., description="The name of the customer")
    product_name: str = Field(..., description="The name of the product")
    rating: int = Field(..., ge=1, le=5, description="The rating given (1-5)")
    review_text: str = Field(..., description="The review text content")


class WriteReviewResponse(BaseModel):
    """Response from the write_review function."""

    success: bool = Field(..., description="Whether the operation was successful")
    message: str = Field(..., description="A message describing the result")
    review: ReviewDetails = Field(..., description="Details of the submitted review")
    note: str = Field(..., description="A note about the mock operation")


class EmailDetails(BaseModel):
    """Details of a sent email."""

    to: str = Field(..., description="The recipient email address")
    cc: str = Field(..., description="The CC email address or 'None'")
    content: str = Field(..., description="The email content")
    timestamp: str = Field(..., description="The timestamp of the email")


class SendEmailResponse(BaseModel):
    """Response from the send_email function."""

    success: bool = Field(..., description="Whether the operation was successful")
    message: str = Field(..., description="A message describing the result")
    email_details: EmailDetails = Field(..., description="Details of the sent email")
    note: str = Field(..., description="A note about the mock operation")


class OrderDetails(BaseModel):
    """Details of a placed order."""

    customer_name: str = Field(..., description="The name of the customer")
    customer_email: str = Field(..., description="The email of the customer")
    product_name: str = Field(..., description="The name of the product")
    product_id: str = Field(..., description="The ID of the product")
    quantity: int = Field(..., description="The quantity ordered")
    unit_price: float = Field(..., description="The unit price of the product")
    total: float = Field(..., description="The total cost of the order")
    new_total_orders: int = Field(..., description="The customer's new total order count")
    new_total_spent: float = Field(..., description="The customer's new total spent amount")


class UpdateCustomerInfoResponse(BaseModel):
    """Response from the update_customer_info function."""

    success: bool = Field(..., description="Whether the operation was successful")
    message: str = Field(..., description="A message describing the result")
    order_details: OrderDetails = Field(..., description="Details of the placed order")
    note: str = Field(..., description="A note about the mock operation")


# ============================================================================
# Configuration
# ============================================================================


class RetailToolsConfig(FunctionGroupBaseConfig, name="retail_tools"):
    """Configuration for the retail agent tools."""

    data_dir: Path = Field(
        default_factory=lambda: Path(__file__).parent / "data",
        description="Directory containing the customer and product JSON files.",
    )
    include: list[str] = Field(
        default_factory=lambda: [
            "get_customer_by_email",
            "get_customer_by_id",
            "get_product_info",
            "get_all_products",
            "write_review",
            "send_email",
            "update_customer_info", ],
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
    # Load data files as typed Pydantic models
    customers_file = _config.data_dir / "customers.json"
    products_file = _config.data_dir / "products.json"

    try:
        with open(customers_file, encoding="utf-8") as f:
            customers_data: list[Customer] = [Customer(**c) for c in json.load(f)]
    except (FileNotFoundError, json.JSONDecodeError) as e:
        raise RuntimeError("Failed to load customers data") from e

    try:
        with open(products_file, encoding="utf-8") as f:
            products_data: list[Product] = [Product(**p) for p in json.load(f)]
    except (FileNotFoundError, json.JSONDecodeError) as e:
        raise RuntimeError("Failed to load products data") from e

    group = FunctionGroup(config=_config)

    async def _get_customer_by_email(email: str) -> Customer:
        """Search for a customer by their email address.

        Args:
            email: The customer's email address.

        Returns:
            Customer information including id, name, past orders, total orders and total spent.
        """
        for customer in customers_data:
            if customer.email.lower() == email.lower():
                return customer

        raise RuntimeError({
            "error": f"No customer found with email: {email}",
            "message": "This appears to be a new customer. They have no purchase history.",
        })

    async def _get_customer_by_id(customer_id: str) -> Customer:
        """Look up a customer by their unique customer ID.

        Args:
            customer_id: The customer's unique identifier (for example CUST001).

        Returns:
            Customer information including id, name, email, past orders, total orders and total spent.
        """
        for customer in customers_data:
            if customer.id == customer_id:
                return customer

        raise RuntimeError({
            "error": f"No customer found with ID: {customer_id}",
            "message": "Please verify the customer ID is correct.",
        })

    async def _get_product_info(product_identifier: str) -> Product:
        """Retrieve detailed information about a specific product.

        Args:
            product_identifier: Either the product ID (for example PROD001).

        Returns:
            Product information including id, name, description, price, stock, and reviews.
        """
        for product in products_data:
            if product.id == product_identifier or product.name.lower() == product_identifier.lower():
                return product

        raise RuntimeError({
            "error": f"No product found with identifier: {product_identifier}",
            "message": "Please verify the product identifier is correct.",
        })

    async def _get_all_products(dummy: str = "") -> list[ProductSummary]:
        """Retrieve a list of all available products.

        Returns:
            List of all products with their basic information (id, name, description, price, stock).
        """
        del dummy
        return [
            ProductSummary(
                id=p.id,
                name=p.name,
                description=p.description,
                price=p.price,
                stock=p.stock,
                average_rating=(sum(r.rating for r in p.reviews) / len(p.reviews) if p.reviews else "No ratings yet"),
                review_count=len(p.reviews),
                review_texts=[r.review for r in p.reviews],
            ) for p in products_data
        ]

    async def _write_review(params: WriteReviewParams) -> WriteReviewResponse:
        """Submit a product review (mock function - does not persist data).

        Args:
            params: WriteReviewParams with customer_email, product_name, rating, and review_text.

        Returns:
            Success confirmation with review details.
        """
        # Check if customer exists (will raise RuntimeError if not found)
        customer = await _get_customer_by_email(params.customer_email)

        # Check if product exists (will raise RuntimeError if not found)
        product = await _get_product_info(params.product_name)

        # Mock success response
        return WriteReviewResponse(
            success=True,
            message=f"Review submitted successfully for {product.name}",
            review=ReviewDetails(
                customer_name=customer.name,
                product_name=product.name,
                rating=params.rating,
                review_text=params.review_text,
            ),
            note="This is a mock operation - the review was not actually saved to the database.",
        )

    async def _send_email(params: SendEmailParams) -> SendEmailResponse:
        """Send an email to a customer (mock function - no actual email sent).

        Args:
            params: SendEmailParams with recipient_email, content, and optional cc.

        Returns:
            Success confirmation with email details.
        """
        return SendEmailResponse(
            success=True,
            message="Email sent successfully",
            email_details=EmailDetails(
                to=params.recipient_email,
                cc=params.cc or "None",
                content=params.content,
                timestamp="2024-11-25T10:00:00Z",
            ),
            note="This is a mock operation - no actual email was sent.",
        )

    async def _update_customer_info(params: UpdateCustomerInfoParams) -> UpdateCustomerInfoResponse:
        """Update customer information with a new order (mock function - does not persist data).

        Args:
            params: UpdateCustomerInfoParams with customer_email, product_name, and quantity.

        Returns:
            Success confirmation with updated order details.
        """
        # Check if customer exists (will raise RuntimeError if not found)
        customer = await _get_customer_by_email(params.customer_email)

        # Check if product exists (will raise RuntimeError if not found)
        product = await _get_product_info(params.product_name)

        # Check stock availability
        if product.stock < params.quantity:
            raise RuntimeError({
                "error": "Insufficient stock",
                "message": f"Only {product.stock} units of {product.name} are available.",
            })

        # Calculate order total
        order_total = product.price * params.quantity

        # Mock success response
        return UpdateCustomerInfoResponse(
            success=True,
            message=f"Order placed successfully for {customer.name}",
            order_details=OrderDetails(
                customer_name=customer.name,
                customer_email=customer.email,
                product_name=product.name,
                product_id=product.id,
                quantity=params.quantity,
                unit_price=product.price,
                total=order_total,
                new_total_orders=customer.total_orders + 1,
                new_total_spent=customer.total_spent + order_total,
            ),
            note="This is a mock operation - the order was not actually saved to the database.",
        )

    # Add functions to the group
    group.add_function(name="get_customer_by_email",
                       fn=_get_customer_by_email,
                       description=_get_customer_by_email.__doc__)
    group.add_function(name="get_customer_by_id", fn=_get_customer_by_id, description=_get_customer_by_id.__doc__)
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
