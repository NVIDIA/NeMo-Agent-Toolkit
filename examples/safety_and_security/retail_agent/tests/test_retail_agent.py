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

import typing
from pathlib import Path

import pytest
import pytest_asyncio

if typing.TYPE_CHECKING:
    from nat.builder.workflow import Workflow


@pytest_asyncio.fixture(name="workflow", scope="module")
async def workflow_fixture():
    """Load the retail agent workflow for testing."""
    from nat.runtime.loader import load_workflow
    from nat.test.utils import locate_example_config
    from nat_retail_agent.register import RetailToolsConfig

    config_file: Path = locate_example_config(RetailToolsConfig)
    async with load_workflow(config_file) as workflow:
        yield workflow


async def run_retail_agent(workflow: "Workflow", email_input: dict[str, str]) -> str:
    """Helper function to run the retail agent with an email input.

    Args:
        workflow: The workflow instance.
        email_input: Dictionary with 'from', 'content', and optionally 'cc' fields.

    Returns:
        The agent's response as a string.
    """
    # Format the email input for the agent
    formatted_input = f"""
Email From: {email_input["from"]}
CC: {email_input.get("cc", "None")}
Content:
{email_input["content"]}
"""
    async with workflow.run(formatted_input) as runner:
        result = await runner.result(to_type=str)
    return result


@pytest.mark.integration
@pytest.mark.usefixtures("nvidia_api_key")
@pytest.mark.asyncio
async def test_product_inquiry(workflow: "Workflow"):
    """Test that the agent can handle product information inquiries."""
    email_input = {
        "from": "david.brown@email.com",
        "content": "Hello, I'm interested in learning about your garden trowels. What do you have available?",
    }

    result = await run_retail_agent(workflow, email_input)
    result_lower = result.lower()

    # Check that the agent mentions the trowel product
    assert any(keyword in result_lower for keyword in ["trowel", "garden trowel", "premium garden"]), (
        f"Expected product information in response, got: {result}")


@pytest.mark.integration
@pytest.mark.usefixtures("nvidia_api_key")
@pytest.mark.asyncio
async def test_review_submission(workflow: "Workflow"):
    """Test that the agent can handle review submissions from existing customers."""
    email_input = {
        "from":
            "john.doe@email.com",
        "content":
            "I'd like to write a review for the Premium Garden Trowel I purchased. It's fantastic! I give it 5 stars.",
    }

    result = await run_retail_agent(workflow, email_input)
    result_lower = result.lower()

    # Check that the agent acknowledges the review submission
    assert any(keyword in result_lower for keyword in ["review", "thank", "submitted", "feedback", "appreciate"]), (
        f"Expected review acknowledgment in response, got: {result}")


@pytest.mark.integration
@pytest.mark.usefixtures("nvidia_api_key")
@pytest.mark.asyncio
async def test_order_placement(workflow: "Workflow"):
    """Test that the agent can handle order placement requests."""
    email_input = {
        "from": "sarah.smith@email.com",
        "content": "I would like to order 2 watering cans. Can you please process this order?",
    }

    result = await run_retail_agent(workflow, email_input)
    result_lower = result.lower()

    # Check that the agent mentions the order or watering can
    assert any(keyword in result_lower for keyword in ["order", "watering can", "purchase", "total", "price"]), (
        f"Expected order information in response, got: {result}")


@pytest.mark.integration
@pytest.mark.usefixtures("nvidia_api_key")
@pytest.mark.asyncio
async def test_customer_history_lookup(workflow: "Workflow"):
    """Test that the agent can look up customer purchase history."""
    email_input = {
        "from": "emma.wilson@email.com",
        "content": "Can you show me my order history?",
    }

    result = await run_retail_agent(workflow, email_input)
    result_lower = result.lower()

    # Check that the agent provides customer history information
    assert any(keyword in result_lower for keyword in ["order", "purchase", "history", "bought", "past orders"]), (
        f"Expected customer history in response, got: {result}")


@pytest.mark.integration
@pytest.mark.usefixtures("nvidia_api_key")
@pytest.mark.asyncio
async def test_product_comparison(workflow: "Workflow"):
    """Test that the agent can compare multiple products."""
    email_input = {
        "from": "mike.johnson@email.com",
        "content": "I'm looking for gardening gloves. Can you show me what options you have and compare them?",
    }

    result = await run_retail_agent(workflow, email_input)
    result_lower = result.lower()

    # Check that the agent provides product information
    assert any(keyword in result_lower
               for keyword in ["glove", "price", "option"]), (f"Expected product comparison in response, got: {result}")
