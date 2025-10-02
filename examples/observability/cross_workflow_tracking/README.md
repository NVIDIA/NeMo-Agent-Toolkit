# Cross-Workflow Observability Example

This example demonstrates how to use NeMo Agent Toolkit's cross-workflow observability feature to track and monitor execution across multiple interconnected workflows. This feature provides end-to-end visibility into complex workflow chains and enables comprehensive performance analysis and debugging.

## Overview

The cross-workflow observability feature allows you to:

- **Propagate observability context** across multiple workflow executions
- **Track parent-child relationships** between workflows with hierarchical depth tracking
- **Maintain trace continuity** across workflow boundaries for distributed processing
- **Analyze performance and dependencies** across workflow chains with timing data
- **Debug issues** that span multiple workflows with comprehensive error tracking
- **Serialize and restore context** for distributed systems and microservices
- **Add custom attributes** for enhanced filtering and analysis

## Key Components

### 1. ObservabilityContext
Manages trace IDs, span hierarchies, and workflow metadata:

```python
from nat.observability import ObservabilityContext

# Create a root context
context = ObservabilityContext.create_root_context("main_workflow")

# Create child context for sub-workflow
child_context = context.create_child_context("sub_workflow")
```

### 2. ObservabilityWorkflowInvoker
Utility for invoking workflows with context propagation:

```python
from nat.observability import ObservabilityWorkflowInvoker

# Invoke workflow with observability context
result = await ObservabilityWorkflowInvoker.invoke_workflow_with_context(
    workflow=my_workflow,
    message=input_data,
    workflow_name="data_processing",
    parent_context=parent_context
)
```

### 3. Cross-Workflow Processors
Enhance spans with cross-workflow information:

```python
from nat.observability import CrossWorkflowProcessor, WorkflowRelationshipProcessor

# Add to your observability configuration
processors = [
    CrossWorkflowProcessor(),
    WorkflowRelationshipProcessor()
]
```

## Usage Patterns

### Pattern 1: Parent-Child Workflows

```python
async def main_workflow(input_data):
    # Create root observability context
    context = ObservabilityContext.create_root_context("main_workflow")

    # Process data in child workflow
    processed_data = await ObservabilityWorkflowInvoker.invoke_workflow_with_context(
        workflow=data_processing_workflow,
        message=input_data,
        workflow_name="data_processing",
        parent_context=context
    )

    # Generate report in another child workflow
    report = await ObservabilityWorkflowInvoker.invoke_workflow_with_context(
        workflow=report_generation_workflow,
        message=processed_data,
        workflow_name="report_generation",
        parent_context=context
    )

    return report
```

### Pattern 2: Sequential Workflow Chain

```python
async def sequential_processing(data):
    context = ObservabilityContext.create_root_context("sequential_processing")

    # Step 1: Validation
    validated_data = await ObservabilityWorkflowInvoker.invoke_workflow_with_context(
        workflow=validation_workflow,
        message=data,
        workflow_name="validation",
        parent_context=context
    )

    # Step 2: Transformation (child of validation)
    current_context = ObservabilityWorkflowInvoker.get_current_observability_context()
    transformed_data = await ObservabilityWorkflowInvoker.invoke_workflow_with_context(
        workflow=transformation_workflow,
        message=validated_data,
        workflow_name="transformation",
        parent_context=current_context
    )

    # Step 3: Storage
    result = await ObservabilityWorkflowInvoker.invoke_workflow_with_context(
        workflow=storage_workflow,
        message=transformed_data,
        workflow_name="storage",
        parent_context=current_context
    )

    return result
```

### Pattern 3: Distributed Workflow Coordination

```python
async def distributed_processing(tasks):
    context = ObservabilityContext.create_root_context("distributed_processing")

    # Process tasks in parallel, each with its own workflow
    results = await asyncio.gather(*[
        ObservabilityWorkflowInvoker.invoke_workflow_with_context(
            workflow=task_processing_workflow,
            message=task,
            workflow_name=f"task_processor_{i}",
            parent_context=context
        )
        for i, task in enumerate(tasks)
    ])

    # Aggregate results
    aggregated = await ObservabilityWorkflowInvoker.invoke_workflow_with_context(
        workflow=aggregation_workflow,
        message=results,
        workflow_name="aggregation",
        parent_context=context
    )

    return aggregated
```

## Observability Data

The cross-workflow observability feature adds the following attributes to spans:

### Trace Information
- `observability.trace_id`: Unique trace identifier across all workflows
- `observability.root_span_id`: Root span of the workflow chain
- `observability.current_span_id`: Current span identifier

### Workflow Information
- `observability.workflow_name`: Name of the current workflow
- `observability.workflow_id`: Unique identifier for the workflow instance
- `observability.workflow_depth`: Nesting level of the workflow
- `observability.workflow_status`: Current status (running, completed, failed)

### Relationship Information
- `relationship.type`: Type of relationship (root_workflow, child_workflow)
- `relationship.parent_workflow_name`: Name of parent workflow
- `relationship.hierarchy_path`: Full path showing workflow chain
- `relationship.nesting_level`: Depth in the workflow hierarchy

### Timing Information
- `observability.workflow_start_time`: Workflow start timestamp
- `observability.workflow_end_time`: Workflow end timestamp
- `observability.workflow_duration`: Execution duration in seconds

## Configuration

To enable cross-workflow observability in your NAT configuration:

```yaml
general:
  telemetry:
    tracing:
      main_exporter:
        type: "file_exporter"
        config:
          file_path: "traces.jsonl"
          processors:
            - type: "cross_workflow_processor"
            - type: "workflow_relationship_processor"
```

## Benefits

1. **End-to-End Visibility**: Track execution flow across multiple workflows
2. **Performance Analysis**: Identify bottlenecks in workflow chains
3. **Dependency Mapping**: Understand workflow relationships and dependencies
4. **Debugging**: Trace issues across workflow boundaries
5. **Compliance**: Maintain audit trails for complex multi-workflow operations

## Best Practices

1. **Use Meaningful Workflow Names**: Choose descriptive names for better observability
2. **Propagate Context Consistently**: Always pass observability context between workflows
3. **Add Custom Attributes**: Include relevant metadata for better filtering and analysis
4. **Monitor Workflow Depth**: Be aware of deeply nested workflows for performance
5. **Handle Errors Gracefully**: Ensure observability context is maintained during error scenarios

## Example Files

This directory contains several example files demonstrating different aspects of cross-workflow observability:

### 1. `example.py`
Basic demonstration of cross-workflow observability concepts with mock workflows:
- Sequential workflow processing with parent-child relationships
- Parallel workflow execution with shared context
- Nested workflow processing with multiple hierarchy levels
- Observability context creation, propagation, and serialization

### 2. `cross_workflow_tracking_example.py`
Real NAT workflow integration demonstrating:
- Actual NAT workflow execution with observability context
- Context setting in workflow's context state
- Real-world usage with customer support scenarios
- Trace ID continuity across multiple queries

### 3. `workflow_integration_example.py`
Advanced integration patterns showing:
- Enhanced workflow execution with observability
- Integration with router agent configurations
- Workflow-to-workflow observability propagation
- Error handling with observability context preservation

### 4. `observability_config.yml`
Sample configuration file showing:
- Telemetry configuration for cross-workflow tracking
- File exporter setup with cross-workflow processors
- LLM and workflow configuration integration

### 5. `integrated_example.py`
Comprehensive integration example demonstrating:
- Full NAT workflow integration with observability context propagation
- Sequential and parallel multi-workflow processing patterns
- Real workflow execution with router agent configurations
- Cross-workflow span processing and enhancement
- Context serialization for distributed processing
- Production-ready error handling and status tracking

## Running the Examples

### Basic Example
```bash
python example.py
```
Demonstrates core concepts with mock workflows.

### Real NAT Integration
```bash
python cross_workflow_tracking_example.py
```
Shows real NAT workflow execution with observability.

### Advanced Integration
```bash
python workflow_integration_example.py
```
Demonstrates advanced patterns with router agent.

### Comprehensive Integration
```bash
python integrated_example.py
```
Shows full NAT workflow integration with complete observability features.

## Example Applications

- **Data Pipeline**: ETL workflows with validation, transformation, and loading stages
- **Document Processing**: OCR → Analysis → Classification → Storage workflows
- **ML Training Pipeline**: Data Prep → Feature Engineering → Training → Evaluation workflows
- **Customer Service**: Request → Routing → Processing → Response workflows
- **Financial Processing**: Validation → Risk Assessment → Processing → Notification workflows
- **Multi-Agent Systems**: Agent coordination with trace continuity across agent interactions