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
Cross-Workflow Observability Example

This example demonstrates how to use NeMo Agent Toolkit's cross-workflow
observability feature to track execution across multiple interconnected workflows.
"""

import asyncio
import time
from typing import Any
from typing import Dict
from typing import Optional

from nat.observability.context import ObservabilityContext


# Mock functions for the example
async def validate_data(data: Dict[str, Any]) -> Dict[str, Any]:
    """Validate input data."""
    print(f"Validating data: {data}")
    # Add validation logic here
    data["validated"] = True
    return data


async def transform_data(data: Dict[str, Any]) -> Dict[str, Any]:
    """Transform validated data."""
    print(f"Transforming data: {data}")
    # Add transformation logic here
    data["transformed"] = True
    data["values"] = [x * 2 for x in data.get("values", [])]
    return data


async def store_data(data: Dict[str, Any]) -> Dict[str, Any]:
    """Store transformed data."""
    print(f"Storing data: {data}")
    # Add storage logic here
    data["stored"] = True
    return data


async def generate_report(data: Dict[str, Any]) -> Dict[str, Any]:
    """Generate a report from stored data."""
    print(f"Generating report from: {data}")
    # Add report generation logic here
    report = {
        "report_id": "report_001",
        "data_summary": f"Processed {len(data.get('values', []))} values",
        "status": "completed",
        "source_data": data
    }
    return report


async def simulate_workflow_execution(workflow_name: str,
                                      input_data: Dict[str, Any],
                                      parent_context: Optional[ObservabilityContext] = None) -> Dict[str, Any]:
    """
    Simulate a workflow execution with observability context.

    Args:
        workflow_name: Name of the workflow
        input_data: Input data for the workflow
        parent_context: Optional parent observability context

    Returns:
        Result of the workflow execution
    """
    # Create or propagate observability context
    if parent_context:
        obs_context = parent_context.create_child_context(workflow_name)
    else:
        obs_context = ObservabilityContext.create_root_context(workflow_name)

    # Set timing information
    current_workflow = obs_context.get_current_workflow()
    if current_workflow:
        current_workflow.start_time = time.time()
        current_workflow.status = "running"

    print(f"🔄 Executing workflow: {workflow_name}")
    print(f"   - Trace ID: {obs_context.trace_id}")
    print(f"   - Workflow depth: {obs_context.get_workflow_depth()}")
    print(f"   - Input: {input_data}")

    # Simulate the actual workflow function based on name
    if workflow_name == "validation" or "validate" in workflow_name:
        result = await validate_data(input_data)
    elif workflow_name == "transformation" or "transform" in workflow_name:
        result = await transform_data(input_data)
    elif workflow_name == "storage" or "store" in workflow_name:
        result = await store_data(input_data)
    elif "report" in workflow_name:
        result = await generate_report(input_data)
    else:
        # Generic processing
        await asyncio.sleep(0.1)  # Simulate work
        result = input_data.copy()
        result[f"{workflow_name}_processed"] = True

    # Update completion status
    if current_workflow:
        current_workflow.end_time = time.time()
        current_workflow.status = "completed"
        if current_workflow.start_time:
            duration = current_workflow.end_time - current_workflow.start_time
            print(f"   - Duration: {duration:.3f}s")

    return result


async def main():
    """Main function demonstrating cross-workflow observability."""

    print("=" * 60)
    print("Cross-Workflow Observability Demonstration")
    print("=" * 60)

    # Example 1: Sequential Processing with Cross-Workflow Observability
    print("\n=== Example 1: Sequential Processing ===")

    # Create root observability context
    root_context = ObservabilityContext.create_root_context("data_processing_pipeline")
    root_context.add_attribute("pipeline_version", "1.0")
    root_context.add_attribute("environment", "development")

    # Input data
    input_data = {"values": [1, 2, 3, 4, 5], "metadata": {"source": "api", "timestamp": "2024-01-01T00:00:00Z"}}

    # Step 1: Validation
    print("\nStep 1: Data Validation")
    validated_data = await simulate_workflow_execution("validation", input_data, root_context)

    # Step 2: Transformation (child of validation)
    print("\nStep 2: Data Transformation")
    validation_context = root_context.create_child_context("validation")
    transformed_data = await simulate_workflow_execution("transformation", validated_data, validation_context)

    # Step 3: Storage
    print("\nStep 3: Data Storage")
    stored_data = await simulate_workflow_execution("storage", transformed_data, validation_context)

    print(f"\nSequential processing result: {stored_data}")

    # Example 2: Parallel Processing with Shared Context
    print("\n=== Example 2: Parallel Processing ===")

    # Create a new root context for parallel processing
    parallel_context = ObservabilityContext.create_root_context("parallel_data_processing")
    parallel_context.add_attribute("processing_mode", "parallel")

    # Multiple data sets to process in parallel
    datasets = [{
        "values": [1, 2, 3], "id": "dataset_1"
    }, {
        "values": [4, 5, 6], "id": "dataset_2"
    }, {
        "values": [7, 8, 9], "id": "dataset_3"
    }]

    # Process datasets in parallel, each with its own workflow chain
    print("\nProcessing datasets in parallel:")
    tasks = []
    for i, dataset in enumerate(datasets):
        task = process_dataset_chain(dataset, i, parallel_context)
        tasks.append(task)

    parallel_results = await asyncio.gather(*tasks)

    # Generate consolidated report
    print("\nGenerating consolidated report:")
    consolidated_report = await simulate_workflow_execution("consolidated_reporting", {
        "results": parallel_results, "processing_mode": "parallel"
    },
                                                            parallel_context)

    print(f"\nParallel processing report: {consolidated_report}")

    # Example 3: Nested Workflow Processing
    print("\n=== Example 3: Nested Workflow Processing ===")

    # Create context for nested processing
    nested_context = ObservabilityContext.create_root_context("nested_processing_pipeline")
    nested_context.add_attribute("complexity", "high")

    # Complex nested processing
    complex_data = {
        "batches": [{
            "values": [10, 20, 30], "batch_id": "batch_1"
        }, {
            "values": [40, 50, 60], "batch_id": "batch_2"
        }]
    }

    nested_result = await process_complex_nested_workflow(complex_data, nested_context)

    print(f"\nNested processing result: {nested_result}")

    print("\n=== Observability Summary ===")
    print("This example demonstrated:")
    print("- observability.trace_id: Unique trace across all workflows")
    print("- observability.workflow_chain: Full workflow execution path")
    print("- relationship.hierarchy_path: Parent-child workflow relationships")
    print("- observability.workflow_depth: Nesting level")
    print("- Custom attributes and timing information")
    print("\nIn a real NAT implementation, this data would be automatically")
    print("captured and sent to observability platforms like Phoenix, Weave, or Langfuse.")


async def process_dataset_chain(dataset: Dict[str, Any], dataset_index: int,
                                parent_context: ObservabilityContext) -> Dict[str, Any]:
    """Process a single dataset through the complete workflow chain."""

    # Create a child context for this dataset processing
    dataset_context = parent_context.create_child_context(f"dataset_processor_{dataset_index}")
    dataset_context.add_attribute("dataset_id", dataset.get("id", f"dataset_{dataset_index}"))

    print(f"\nProcessing dataset {dataset_index + 1}: {dataset.get('id', f'dataset_{dataset_index}')}")

    # Validation
    validated = await simulate_workflow_execution(f"validation_{dataset_index}", dataset, dataset_context)

    # Transformation
    transformed = await simulate_workflow_execution(f"transformation_{dataset_index}", validated, dataset_context)

    # Storage
    stored = await simulate_workflow_execution(f"storage_{dataset_index}", transformed, dataset_context)

    return stored


async def process_complex_nested_workflow(data: Dict[str, Any], parent_context: ObservabilityContext) -> Dict[str, Any]:
    """Process complex nested workflow with multiple levels of nesting."""

    # Create context for batch processing
    batch_context = parent_context.create_child_context("batch_processor")
    batch_context.add_attribute("total_batches", len(data.get("batches", [])))

    processed_batches = []

    print(f"\nProcessing {len(data.get('batches', []))} batches in nested workflow:")

    # Process each batch
    for i, batch in enumerate(data.get("batches", [])):
        # Create context for individual batch
        individual_batch_context = batch_context.create_child_context(f"batch_{i}")
        individual_batch_context.add_attribute("batch_id", batch.get("batch_id"))

        print(f"\nBatch {i + 1}: {batch.get('batch_id')}")

        # Process batch through sub-workflows
        batch_result = await process_dataset_chain(batch, i, individual_batch_context)

        processed_batches.append(batch_result)

    # Generate batch summary report
    print("\nGenerating batch summary report:")
    batch_summary = await simulate_workflow_execution(
        "batch_summary_report", {
            "batches": processed_batches, "total_count": len(processed_batches)
        }, batch_context)

    return batch_summary


if __name__ == "__main__":
    asyncio.run(main())
