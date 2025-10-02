#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
NAT Workflow Integration with Cross-Workflow Observability

This example shows how to integrate cross-workflow observability with
real NAT workflows using the enhanced workflow.run() method.
"""

import asyncio
import logging
import os

from nat.observability.context import ObservabilityContext
from nat.runtime.loader import load_workflow

logger = logging.getLogger(__name__)


async def enhanced_workflow_execution():
    """Demonstrate enhanced workflow execution with observability."""

    print("=" * 60)
    print("NAT Workflow + Cross-Workflow Observability Integration")
    print("=" * 60)

    # Create root observability context
    root_context = ObservabilityContext.create_root_context("multi_stage_assistant")
    root_context.add_attribute("session_id", "demo_001")
    root_context.add_attribute("user", "developer")

    # Path to the router agent config (if available)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    router_config = os.path.join(current_dir, "../../control_flow/router_agent/configs/config.yml")

    if not os.path.exists(router_config):
        print("Router agent config not found. Creating demo workflow...")
        await demo_with_mock_workflow(root_context)
        return

    print(f"Using router agent config: {router_config}")

    # Example queries to process
    queries = [
        "What yellow fruit would you recommend?",
        "Can you suggest a good city to visit?",
        "What's a good book to read?"
    ]

    try:
        # Load the workflow once
        async with load_workflow(router_config) as workflow:

            for i, query in enumerate(queries):
                print(f"\n--- Query {i+1}: {query} ---")

                # Create child context for this query
                query_context = root_context.create_child_context(f"query_processing_{i+1}")
                query_context.add_attribute("query_type", "user_request")
                query_context.add_attribute("query_index", i + 1)

                # Set observability context in the workflow's context state
                workflow._context_state.observability_context.set(query_context)

                # Execute workflow
                async with workflow.run(query) as runner:
                    result = await runner.result(to_type=str)

                print(f"Result: {result}")

                # Demonstrate context information
                current_workflow = query_context.get_current_workflow()
                if current_workflow:
                    print(f"Workflow depth: {query_context.get_workflow_depth()}")
                    print(f"Trace ID: {query_context.trace_id}")

    except Exception as e:
        print(f"Error in workflow execution: {e}")
        await demo_with_mock_workflow(root_context)


async def demo_with_mock_workflow(root_context: ObservabilityContext):
    """Demo with mock workflow when router agent is not available."""

    print("\nDemo with mock workflow execution:")

    # Simulate multiple workflow stages
    stages = [("input_validation", "Validating user input"), ("intent_classification", "Classifying user intent"),
              ("response_generation", "Generating response"), ("quality_check", "Performing quality check")]

    result = {"input": "What's a good programming language to learn?"}

    for stage_name, description in stages:
        print(f"\n🔄 {description}")

        # Create child context for this stage
        stage_context = root_context.create_child_context(stage_name)
        stage_context.add_attribute("stage_type", "processing")
        stage_context.add_attribute("description", description)

        # Simulate processing
        await asyncio.sleep(0.2)

        # Update result
        result[stage_name] = f"Completed {stage_name}"

        print(f"   - Stage: {stage_name}")
        print(f"   - Trace ID: {stage_context.trace_id}")
        print(f"   - Depth: {stage_context.get_workflow_depth()}")
        print(f"   - Parent: {stage_context.workflow_chain[-1].parent_workflow_id}")

    print(f"\nFinal result: {result}")


async def demonstrate_workflow_with_steps():
    """Demonstrate workflow execution with intermediate steps and observability."""

    print("\n" + "=" * 60)
    print("Workflow with Steps + Observability")
    print("=" * 60)

    current_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(current_dir, "observability_config.yml")

    # Create observability context
    steps_context = ObservabilityContext.create_root_context("workflow_with_steps")
    steps_context.add_attribute("execution_mode", "with_steps")
    steps_context.add_attribute("tracking_enabled", True)

    try:
        # Load workflow with observability config
        async with load_workflow(config_path) as workflow:
            # Set observability context in the workflow's context state
            workflow._context_state.observability_context.set(steps_context)

            # Execute workflow (SessionManager doesn't have result_with_steps method)
            async with workflow.run("What programming language should I learn?") as runner:
                result = await runner.result(to_type=str)
                steps = []  # SessionManager doesn't provide intermediate steps

            print(f"Result: {result}")
            print(f"Number of intermediate steps: {len(steps) if steps else 0}")

            # Show observability context info
            print(f"Trace ID: {steps_context.trace_id}")
            print(f"Workflow chain: {[w.workflow_name for w in steps_context.workflow_chain]}")

    except FileNotFoundError:
        print(f"Config file not found: {config_path}")
        print("Using mock demonstration...")

        # Mock the workflow execution with steps
        await demo_with_mock_workflow(steps_context)


async def main():
    """Main function demonstrating the integration."""

    # Set up logging
    logging.basicConfig(level=logging.INFO)

    # Example 1: Enhanced workflow execution
    await enhanced_workflow_execution()

    # Example 2: Workflow with steps
    await demonstrate_workflow_with_steps()

    print("\n" + "=" * 60)
    print("Integration Complete!")
    print("=" * 60)
    print("\nKey Features Demonstrated:")
    print("✅ Observability context integration with workflow execution")
    print("✅ Observability context propagation through workflow execution")
    print("✅ Automatic context propagation through workflow execution")
    print("✅ Trace ID continuity across workflow stages")
    print("✅ Custom attributes and metadata tracking")
    print("✅ Workflow hierarchy and depth tracking")

    print("\nNext Steps:")
    print("1. Add cross-workflow processors to your telemetry configuration")
    print("2. Configure observability exporters (Phoenix, Weave, Langfuse)")
    print("3. Use ObservabilityWorkflowInvoker for workflow-to-workflow calls")
    print("4. Monitor traces in your observability platform")


if __name__ == "__main__":
    asyncio.run(main())
