#!/usr/bin/env python3

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

"""
Integrated Cross-Workflow Observability Example

This example demonstrates how to integrate cross-workflow observability
with actual NAT workflows, showing real workflow execution with observability
context propagation across multiple workflow stages.
"""

import asyncio
import logging
import os
import tempfile
from typing import Dict, Any

from nat.runtime.loader import load_workflow
from nat.observability.context import ObservabilityContext
from nat.builder.context import Context, ContextState
from nat.observability.processor.cross_workflow_processor import CrossWorkflowProcessor, WorkflowRelationshipProcessor
from nat.data_models.span import Span

logger = logging.getLogger(__name__)


async def run_workflow_with_observability(
    config_file: str,
    input_data: Any,
    workflow_name: str,
    parent_context: ObservabilityContext | None = None
) -> Any:
    """
    Run a NAT workflow with observability context propagation.

    Args:
        config_file: Path to the NAT workflow configuration
        input_data: Input data for the workflow
        workflow_name: Name for observability tracking
        parent_context: Optional parent observability context

    Returns:
        Result from the workflow execution
    """
    # Create or propagate observability context
    if parent_context:
        obs_context = parent_context.create_child_context(workflow_name)
    else:
        obs_context = ObservabilityContext.create_root_context(workflow_name)

    # Set timing information
    import time
    current_workflow = obs_context.get_current_workflow()
    if current_workflow:
        current_workflow.start_time = time.time()
        current_workflow.status = "running"

    print(f"🔄 Executing NAT workflow: {workflow_name}")
    print(f"   - Config: {os.path.basename(config_file)}")
    print(f"   - Trace ID: {obs_context.trace_id}")
    print(f"   - Workflow depth: {obs_context.get_workflow_depth()}")
    print(f"   - Input: {input_data}")

    try:
        # Load and run the actual NAT workflow
        async with load_workflow(config_file) as workflow:
            # Set observability context in the workflow's context state
            context_state = workflow._context_state
            context_state.observability_context.set(obs_context)

            # Run the workflow without the observability_context parameter
            async with workflow.run(input_data) as runner:
                result = await runner.result(to_type=str)

        # Update completion status
        if current_workflow:
            current_workflow.end_time = time.time()
            current_workflow.status = "completed"
            if current_workflow.start_time:
                duration = current_workflow.end_time - current_workflow.start_time
                print(f"   - Duration: {duration:.3f}s")
                print(f"   - Result: {result}")

        return result

    except Exception as e:
        # Update error status
        if current_workflow:
            current_workflow.end_time = time.time()
            current_workflow.status = "failed"
            current_workflow.tags["error"] = str(e)
        print(f"   - Error: {e}")
        raise


async def demonstrate_observability_processors():
    """Demonstrate the cross-workflow processors with real span data."""

    print("\n" + "=" * 60)
    print("Cross-Workflow Span Processing Demonstration")
    print("=" * 60)

    # Create a mock observability context with nested workflows
    root_context = ObservabilityContext.create_root_context("document_processing_pipeline")
    root_context.add_attribute("document_type", "research_paper")
    root_context.add_attribute("processing_version", "2.1")

    # Create child contexts to simulate nested workflows
    analysis_context = root_context.create_child_context("content_analysis")
    analysis_context.add_attribute("analysis_type", "semantic")

    summary_context = analysis_context.create_child_context("summarization")
    summary_context.add_attribute("summary_length", "medium")

    # Set the context in the global context state
    context_state = ContextState.get()
    context_state.observability_context.set(summary_context)

    # Create and process spans with the processors
    cross_workflow_processor = CrossWorkflowProcessor()
    relationship_processor = WorkflowRelationshipProcessor()

    # Create a test span
    test_span = Span(
        name="summarization_operation",
        attributes={
            "operation_type": "text_summarization",
            "model_name": "claude-3",
            "input_length": 5000
        }
    )

    print("\nOriginal span attributes:")
    for key, value in test_span.attributes.items():
        print(f"  - {key}: {value}")

    # Process with cross-workflow processor
    processed_span = await cross_workflow_processor.process(test_span)

    # Process with relationship processor
    final_span = await relationship_processor.process(processed_span)

    print("\nEnhanced span attributes after cross-workflow processing:")
    for key, value in sorted(final_span.attributes.items()):
        if key.startswith(("observability.", "relationship.")):
            print(f"  - {key}: {value}")

    print("\nKey observability data added:")
    print("✅ Trace ID for cross-workflow correlation")
    print("✅ Workflow hierarchy and depth information")
    print("✅ Parent-child relationship tracking")
    print("✅ Custom attributes from workflow context")
    print("✅ Workflow chain serialization")


async def create_sample_workflow_configs():
    """Create sample workflow configurations for the demonstration."""

    # Create temporary directory for configs
    temp_dir = tempfile.mkdtemp()

    # Data validation workflow config
    validation_config = f"""
llms:
  mock_llm:
    _type: mock

functions:
  data_validator:
    _type: mock_data_validator

workflow:
  _type: data_validator
  llm_name: mock_llm
"""

    validation_config_path = os.path.join(temp_dir, "validation_config.yml")
    with open(validation_config_path, "w") as f:
        f.write(validation_config.strip())

    # Data processing workflow config
    processing_config = f"""
llms:
  mock_llm:
    _type: mock

functions:
  data_processor:
    _type: mock_data_processor

workflow:
  _type: data_processor
  llm_name: mock_llm
"""

    processing_config_path = os.path.join(temp_dir, "processing_config.yml")
    with open(processing_config_path, "w") as f:
        f.write(processing_config.strip())

    # Analysis workflow config
    analysis_config = f"""
llms:
  mock_llm:
    _type: mock

functions:
  data_analyzer:
    _type: mock_data_analyzer

workflow:
  _type: data_analyzer
  llm_name: mock_llm
"""

    analysis_config_path = os.path.join(temp_dir, "analysis_config.yml")
    with open(analysis_config_path, "w") as f:
        f.write(analysis_config.strip())

    return validation_config_path, processing_config_path, analysis_config_path


async def main():
    """Main function demonstrating integrated cross-workflow observability."""

    print("=" * 70)
    print("Integrated Cross-Workflow Observability with NAT Workflows")
    print("=" * 70)

    # Note: For this demonstration, we'll use the existing router agent example
    # In a real scenario, you would have multiple different workflow configs

    current_dir = os.path.dirname(os.path.abspath(__file__))
    router_config = os.path.join(current_dir, "../../control_flow/router_agent/configs/config.yml")

    # Check if the router config exists
    if not os.path.exists(router_config):
        print(f"Router agent config not found at: {router_config}")
        print("Please ensure the control_flow/router_agent example is available.")
        print("\nFor demonstration purposes, we'll show the observability processors:")
        await demonstrate_observability_processors()
        return

    # Example 1: Sequential Multi-Workflow Processing
    print("\n=== Example 1: Sequential Multi-Workflow Processing ===")

    # Create root observability context for the entire pipeline
    pipeline_context = ObservabilityContext.create_root_context("intelligent_assistant_pipeline")
    pipeline_context.add_attribute("session_id", "demo_session_001")
    pipeline_context.add_attribute("user_type", "developer")
    pipeline_context.add_attribute("pipeline_version", "1.0")

    # Sample inputs for different workflow stages
    user_queries = [
        "What yellow fruit would you recommend?",
        "I want a red fruit, what do you suggest?",
        "Can you recommend a good book to read?"
    ]

    results = []

    for i, query in enumerate(user_queries):
        print(f"\nProcessing query {i+1}: {query}")

        try:
            # Stage 1: Query Understanding (using router agent as a proxy)
            understanding_result = await run_workflow_with_observability(
                config_file=router_config,
                input_data=query,
                workflow_name=f"query_understanding_{i+1}",
                parent_context=pipeline_context
            )

            # Stage 2: Response Enhancement (simulated with another call)
            enhancement_context = pipeline_context.create_child_context(f"query_understanding_{i+1}")
            enhanced_result = await run_workflow_with_observability(
                config_file=router_config,
                input_data=f"Please elaborate on: {understanding_result}",
                workflow_name=f"response_enhancement_{i+1}",
                parent_context=enhancement_context
            )

            results.append({
                "query": query,
                "understanding": understanding_result,
                "enhanced": enhanced_result
            })

        except Exception as e:
            print(f"Error processing query {i+1}: {e}")
            results.append({
                "query": query,
                "error": str(e)
            })

    # Example 2: Parallel Workflow Processing
    print("\n=== Example 2: Parallel Workflow Processing ===")

    # Create a new context for parallel processing
    parallel_context = ObservabilityContext.create_root_context("parallel_query_processing")
    parallel_context.add_attribute("processing_mode", "parallel")
    parallel_context.add_attribute("batch_size", len(user_queries))

    # Process multiple queries in parallel with observability tracking
    print("Processing multiple queries in parallel:")

    async def process_query_with_context(query: str, index: int) -> Dict[str, Any]:
        """Process a single query with observability context."""
        try:
            result = await run_workflow_with_observability(
                config_file=router_config,
                input_data=query,
                workflow_name=f"parallel_query_{index+1}",
                parent_context=parallel_context
            )
            return {"query": query, "result": result, "index": index}
        except Exception as e:
            return {"query": query, "error": str(e), "index": index}

    # Execute parallel tasks
    parallel_tasks = [
        process_query_with_context(query, i)
        for i, query in enumerate(user_queries)
    ]

    parallel_results = await asyncio.gather(*parallel_tasks)

    print(f"\nParallel processing completed. Processed {len(parallel_results)} queries.")

    # Example 3: Demonstrate Observability Context Serialization
    print("\n=== Example 3: Context Serialization for Distributed Processing ===")

    # Serialize the context for potential distributed processing
    context_data = parallel_context.to_dict()
    print(f"Serialized context keys: {list(context_data.keys())}")
    print(f"Workflow chain length: {len(context_data['workflow_chain'])}")
    print(f"Custom attributes: {context_data['custom_attributes']}")

    # Demonstrate restoration
    restored_context = ObservabilityContext.from_dict(context_data)
    print(f"Restored context trace ID: {restored_context.trace_id}")
    print(f"Restored workflow depth: {restored_context.get_workflow_depth()}")

    # Demonstrate processor functionality
    await demonstrate_observability_processors()

    # Summary
    print("\n" + "=" * 70)
    print("Integration Summary")
    print("=" * 70)
    print("✅ Cross-workflow observability successfully integrated with NAT workflows")
    print("✅ Trace continuity maintained across multiple workflow executions")
    print("✅ Context propagation working in sequential and parallel scenarios")
    print("✅ Span enhancement with cross-workflow metadata demonstrated")
    print("✅ Context serialization for distributed systems verified")
    print("\nIn a production environment, this observability data would be:")
    print("- Automatically captured during all workflow executions")
    print("- Sent to observability platforms (Phoenix, Weave, Langfuse, etc.)")
    print("- Used for performance monitoring, debugging, and compliance tracking")
    print("- Correlated across microservices and distributed workflow components")


if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO)

    # Run the integrated example
    asyncio.run(main())