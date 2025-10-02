#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Simple Real Cross-Workflow Observability Example

Uses actual NAT workflows with cross-workflow observability.
"""

import asyncio
import os
import tempfile

from nat.observability.context import ObservabilityContext
from nat.runtime.loader import load_workflow


async def create_simple_config() -> str:
    """Create a simple workflow config using built-in NAT functions."""

    config_content = """
llms:
  demo_llm:
    _type: nat_test_llm
    response_seq:
      - "Stubbed workflow reply."
    delay_ms: 0
workflow:
  _type: chat_completion
  llm_name: demo_llm
  system_prompt: "You are a helpful customer support assistant. Provide clear, concise, and helpful responses."
"""

    # Create temporary config file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yml', delete=False) as f:
        f.write(config_content.strip())
        return f.name


async def main() -> None:
    """Demonstrate real NAT workflow with cross-workflow observability."""

    print("=" * 50)
    print("Real NAT Workflow + Cross-Workflow Observability")
    print("=" * 50)

    # Create config file
    config_path = await create_simple_config()

    try:
        # Create root observability context
        root_context = ObservabilityContext.create_root_context("customer_support_pipeline")
        root_context.add_attribute("session_id", "demo_123")
        root_context.add_attribute("user", "customer")

        print("\nRoot context created:")
        print(f"  - Trace ID: {root_context.trace_id}")
        print(f"  - Workflow: {root_context.workflow_chain[0].workflow_name}")

        # Sample customer queries
        queries = [
            "What are your business hours?", "How can I return a product?", "What payment methods do you accept?"
        ]

        # Load the workflow
        async with load_workflow(config_path) as workflow:

            for i, query in enumerate(queries, 1):
                print(f"\n--- Processing Query {i} ---")
                print(f"Query: {query}")

                # Create child context for this query
                query_context = root_context.create_child_context(f"query_handler_{i}")
                query_context.add_attribute("query_type", "customer_inquiry")
                query_context.add_attribute("priority", "normal")

                # Set observability context in the workflow's context state
                workflow._context_state.observability_context.set(query_context)

                # Execute workflow
                async with workflow.run(query) as runner:
                    result = await runner.result(to_type=str)

                print(f"Response: {result}")
                print("Observability Info:")
                print(f"  - Trace ID: {query_context.trace_id}")
                print(f"  - Workflow Depth: {query_context.get_workflow_depth()}")
                print(f"  - Parent Workflow: {query_context.workflow_chain[0].workflow_name}")
                print(f"  - Current Workflow: {query_context.workflow_chain[-1].workflow_name}")

        # Demonstrate context serialization
        print("\n--- Context Serialization ---")
        context_data = root_context.to_dict()
        print(f"Serialized context has {len(context_data)} keys")
        print(f"Workflow chain length: {len(context_data['workflow_chain'])}")
        print(f"Custom attributes: {context_data['custom_attributes']}")

        # Restore context
        restored_context = ObservabilityContext.from_dict(context_data)
        print(f"Restored trace ID matches: {restored_context.trace_id == root_context.trace_id}")

    finally:
        # Clean up config file
        os.unlink(config_path)

    print(f"\n{'='*50}")
    print("✅ Real workflow integration successful!")
    print("Key features demonstrated:")
    print("- Real NAT workflow execution with observability")
    print("- Trace ID continuity across multiple queries")
    print("- Parent-child workflow relationships")
    print("- Custom attributes and metadata")
    print("- Context serialization for distributed systems")


if __name__ == "__main__":
    asyncio.run(main())
