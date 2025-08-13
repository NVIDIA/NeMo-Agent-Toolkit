<!--
SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# Adding Unit Tests for Tools

## Overview

Use `nat.test.ToolTestRunner` to test tools in complete isolation without requiring spinning up entire workflows, agents, and external services. This allows you to validate tool functionality quickly and reliably during development.

## Basic Usage

### Testing a Simple Tool

The following example demonstrates testing a basic multiplication tool:

```python
from nat.test import ToolTestRunner
from my_calculator.register import MultiplyToolConfig

async def test_multiply_tool():
    runner = ToolTestRunner()
    result = await runner.test_tool(
        config_type=MultiplyToolConfig,
        input_data="What is 2 times 4?",
        expected_output="The product of 2 * 4 is 8"
    )

    # The framework automatically validates the expected output
    # Add additional assertions if needed
    assert "8" in result
    assert "product" in result
```

### Testing Error Handling

Verify that your tools handle invalid input:

```python
async def test_tool_error_handling():
    runner = ToolTestRunner()
    result = await runner.test_tool(
        config_type=MultiplyToolConfig,
        input_data="Multiply just one number: 5"
    )

    # Tool should return error message for invalid input
    assert "Provide at least 2 numbers" in result
```

## Advanced Usage

### Testing Tools with Dependencies

For tools that depend on LLMs, memory, retrievers, or other components, use the mocked dependencies context:

```python
from nat.test import with_mocked_dependencies

async def test_tool_with_llm_dependency():
    async with with_mocked_dependencies() as (runner, mock_builder):
        # Mock the LLM response
        mock_builder.mock_llm("gpt-4", "Mocked LLM response")

        # Mock memory responses
        mock_builder.mock_memory_client("user_memory", {
            "retrieved_data": "important context"
        })

        # Mock retriever responses
        mock_builder.mock_retriever("knowledge_base", [
            {"text": "relevant document", "score": 0.9}
        ])

        # Test the tool with mocked dependencies
        result = await runner.test_tool_with_builder(
            config_type=SmartToolConfig,
            builder=mock_builder,
            config_params={"llm_name": "gpt-4"},
            input_data="complex query requiring context"
        )

        assert "mocked" in result.lower()
```

### Available Mock Methods

The `MockBuilder` provides mocking for all major components:

```python
# Mock LLM responses
mock_builder.mock_llm("model_name", "Fixed response")

# Mock embedder responses
mock_builder.mock_embedder("embedder_name", [0.1, 0.2, 0.3])

# Mock memory client responses
mock_builder.mock_memory_client("memory_name", {"key": "value"})

# Mock retriever responses
mock_builder.mock_retriever("retriever_name", [
    {"text": "doc1", "score": 0.9},
    {"text": "doc2", "score": 0.8}
])

# Mock function responses
mock_builder.mock_function("function_name", "function result")
```

## Troubleshooting
The following are common errors and their troubleshooting solutions.

### Tool Not Found Error

**Error message**:
```
ValueError: Tool MyToolConfig is not registered. Make sure it's imported and registered with @register_function.
```

**Solution**: Ensure your tool's module is imported before testing:

```python
# Import the module containing your tool registration
import my_package.register  # This registers the tool

from my_package.register import MyToolConfig
```

### Mock Not Working

If mocked dependencies are not being used, check your setup order.

**Incorrect approach**:
```python
# ❌ Wrong: Mock after testing
mock_builder.mock_llm("gpt-4", "response")
result = await runner.test_tool_with_builder(...)
```

**Correct approach**:
```python
# ✅ Correct: Mock before testing
async with with_mocked_dependencies() as (runner, mock_builder):
    mock_builder.mock_llm("gpt-4", "response")  # Mock first
    result = await runner.test_tool_with_builder(
        config_type=MyToolConfig,
        builder=mock_builder,  # Pass the builder
        input_data="test"
    )
```
