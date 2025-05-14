<!--
SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# Writing Custom Functions

Functions can be created in several ways:

* **From a callable**:

    ```python
    # Create a function from a callable
    async def my_function(input_data: MyInputModel) -> MyOutputModel:
        # Process input_data
        return result

    # Create a function info
    function_info = FunctionInfo.from_fn(
        my_function,
        description="My function description"
    )

    # Create a lambda function
    my_function = LambdaFunction.from_info(
        config=MyFunctionConfig(),
        info=function_info
    )
    ```

* **By deriving from the {py:class}`~aiq.builder.function.Function` class**:

    ```python
    class MyCustomFunction(Function[MyInput, MyStreamingOutput, MySingleOutput]):
        def __init__(self, config: MyFunctionConfig):
            super().__init__(
                config=config,
                description="My function description"
            )

        async def _ainvoke(self, value: MyInput) -> MySingleOutput:
            # Implement single output logic
            return result

        async def _astream(self, value: MyInput) -> AsyncGenerator[MyStreamingOutput]:
            # Implement streaming logic
            for item in process(value):
                yield item

    my_function = MyCustomFunction(config=MyFunctionConfig())
    ```

Both of these methods will result in a function that can be used in the same way. The only difference is that the first method is more concise and the second method is more flexible.

## Registering Functions

### Function Configuration Object

To use a function from an AIQ Toolkit configuration file, it must be registered with AIQ Toolkit. Registering a function is done with the {py:deco}`aiq.cli.register_workflow.register_function` decorator. More information about registering components can be found in the [Plugin System](../extend/plugins.md) documentation.

When registering a function, we first need to define the function configuration object. This object is used to configure the function and is passed to the function when it is invoked. Any options that are available to the function must be specified in the configuration object.

An example of a function configuration object is shown below:

```python
class MyFunctionConfig(FunctionBaseConfig, name="my_function"):
    # Sample configuration options
    greeting: str
    option2: int
    option3: dict[str, float]
```

The configuration object must inherit from {py:class}`~aiq.data_models.function.FunctionBaseConfig` and must have a `name` attribute. The `name` attribute is used to identify the function in the configuration file.

Additionally, the configuration object can use Pydantic's features to provide validation and documentation for each of the options. For example, the following configuration will validate that `option2` is a positive integer, and documents all properties with a description and default value.

```python
class MyFunctionConfig(FunctionBaseConfig, name="my_function"):
    greeting: str = Field("Hello from my_custom_workflow workflow!",
                          description="Greeting to respond with")
    option2: int = Field(10, description="Another sample option", ge=0)
    option3: dict[str, float] = Field(default_factory=dict,
                                      description="A dictionary of floats")
```

This additional metadata will ensure that the configuration object is properly validated and the descriptions can be seen when using `aiq info`.

### Function Registration

With the configuration object defined, there are several options available to register the function:

* **Register a function from a callable using {py:class}`~aiq.builder.function_info.FunctionInfo`**:

    ```python
    @register_function(config_type=MyFunctionConfig)
    async def my_function(config: MyFunctionConfig, builder: Builder):

        async def _response_fn(input_message: str) -> str:
            # Process the input_message and generate output.
            # You can access the configuration options here.
            output_message = f"{config.greeting} You said: {input_message}"
            return output_message

        # Yield the function info object which will be used to create a function
        yield FunctionInfo.from_fn(
            _response_fn,
            description="My function description"
        )
    ```

* **Register a function directly from a callable**:

    For simple use cases, you can yield the function directly from the coroutine as shown below:

    ```python
    @register_function(config_type=MyFunctionConfig)
    async def my_function(config: MyFunctionConfig, builder: Builder):

        # Implement your function logic here
        async def _response_fn(input_message: str) -> str:
            """
            My function description
            """

            # Process the input_message and generate output
            output_message = f"Hello from my_custom_workflow workflow! You said: {input_message}"
            return output_message

        # Return the function directly
        yield _response_fn
    ```

    This is functionally equivalent to the first example but is more concise, pulling the description from the docstring.

* **Register a function derived from {py:class}`~aiq.builder.function.Function`**:

    This method is useful when you need to create a function that is more complex than a simple coroutine. For example, you may need to create a function which derives from another function, or one that needs to share state between invocations. In this case, you can create the function instance directly in the register function and yield it.

    ```python
    @register_function(config_type=MyFunctionConfig)
    async def my_function(config: MyFunctionConfig, builder: Builder):

        # Create a class that derives from Function
        class MyCustomFunction(Function[MyInput, NoneType, MySingleOutput]):
            def __init__(self, config: MyFunctionConfig):
                super().__init__(config=config)

            async def _ainvoke(self, value: MyInput) -> MySingleOutput:
                # Implement single output logic
                return result

        yield MyCustomFunction(config=config)
    ```

    :::{note}
    It's important to note that the class is intentionally defined _inside_ of the `my_function` registered coroutine. This is to prevent the class from being created unless the function is going to be instantiated. If the class is defined outside of the coroutine, all of the functions imports will be loaded and the class will be constructed, even if the function is not going to be created. To avoid this, the body of the function must be defined or imported inside of the register function.

    For a more natural syntax, classes can be defined in a separate module and imported into the coroutine as shown below:

    ```python
    @register_function(config_type=MyFunctionConfig)
    async def my_function(config: MyFunctionConfig, builder: Builder):

        # Import the class inside the coroutine
        from my_module import MyCustomFunction

        yield MyCustomFunction(config=config)
    ```

    This also works for callables as shown below:

    ```python
    @register_function(config_type=MyFunctionConfig)
    async def my_function(config: MyFunctionConfig, builder: Builder):

        # Import the callable inside the coroutine
        from my_module import my_callable

        yield my_callable
    ```
    :::

## Initialization and Cleanup

Its required to use an async context manager coroutine to register a function (it's not necessary to use `@asynccontextmanager`, since {py:deco}`aiq.cli.register_workflow.register_function` does this for you). This is because the function may need to execute some initialization before construction or cleanup after it is used. For example, if the function needs to load a model, connect to a resource, or download data, this can be done in the register function.

```python
@register_function(config_type=MyFunctionConfig)
async def my_function(config: MyFunctionConfig, builder: Builder):

    # Perform any initialization logic here such as downloading data
    # Async methods can be used in the register function
    downloaded_data = await download_data()

    # You can also use context managers to manage resources
    async with get_database_connection() as database_connection:

        # Define the function inside of the context manager
        async def _my_function(input_data: MyInput) -> MySingleOutput:

            # Use the database connection with the input data
            result = await database_connection.query(input_data)

            return result

        yield my_callable

    # The database connection will be cleaned up when the context manager is exited

    # Perform any cleanup logic here
    await cleanup_resources()
```

## Input and Output Types

Functions can have any input and output types but are restricted to a single input argument.

### Input Type

The input type is determined in one of two ways:
- When deriving from {py:class}`~aiq.builder.function.Function`, the input type is specified as a generic parameter.
- When creating a function from a callable, the input type is inferred from the callable's signature.
  - If the callable is not annotated with types, an error will be raised.

For example, the following function has an input type of `str`:

```python
class MyFunction(Function[str, NoneType, MySingleOutput]):
    pass
```

And the following function has an input type of `MyCustomClass`:

```python
@register_function(config_type=MyFunctionConfig)
async def my_function(config: MyFunctionConfig, builder: Builder):

    async def _my_function(input_data: MyCustomClass) -> MySingleOutput:
        # Implement the function logic
        return result

    yield FunctionInfo.from_fn(
        _my_function,
        description="My function description"
    )
```

### Output Types

Functions can have two different output types:
- A single output type
  - When the function is invoked with the {py:meth}`~aiq.builder.function.Function.ainvoke` method
- A streaming output type
  - When the function is invoked with the {py:meth}`~aiq.builder.function.Function.astream` method

The output types are determined in one of two ways (identical to the input types):
- When deriving from {py:class}`~aiq.builder.function.Function`, the output types are specified as generic parameters.
- When creating from a callable, the output types are determined from the callable's signature.
  - If the callable is not annotated with types, an error will be raised.

For example, the following function has a single output type of `str`, and no streaming output type:

```python
class MyFunction(Function[MyInput, NoneType, str]):
    pass
```

And the following function has a streaming output type of `MyStreamingOutput`, and no single output type:

```python
@register_function(config_type=MyFunctionConfig)
async def my_function(config: MyFunctionConfig, builder: Builder):

    async def _my_function(input_data: MyInput) -> AsyncGenerator[MyStreamingOutput]:
        # Implement the function logic
        for i in range(10):
            yield MyStreamingOutput(i)

    yield FunctionInfo.from_fn(
        _my_function,
        description="My function description"
    )
```

### Functions with Multiple Arguments

It is possible to create a function with a callable that has multiple arguments. When a function with multiple arguments is passed to {py:meth}`~aiq.builder.function_info.FunctionInfo.from_fn`, the function will be wrapped with a lambda function which takes a single argument and passes it to the original function. For example, the following function takes two arguments, `input_data` and `repeat`:

```python
async def multi_arg_function(input_data: list[float], repeat: int) -> list[float]:
    return [item * repeat for item in input_data]

# Create a function info
function_info = FunctionInfo.from_fn(multi_arg_function)

# Print the input schema
print(function_info.input_schema)
```

This will result in the following input schema:

```python
class MultiArgFunctionInput(BaseModel):
    input_data: list[float]
    repeat: int
```

To invoke the function, input can be passed as a dictionary to the {py:meth}`~aiq.builder.function.Function.ainvoke` method as shown below:

```python
result = await function.ainvoke({"input_data": [1, 2, 3], "repeat": 2})
```

### Supporting Streaming and Single Outputs Simultaneously

It is possible to create a function that supports both streaming and single outputs. When deriving from {py:class}`~aiq.builder.function.Function` implement both {py:meth}`~aiq.builder.function.Function._ainvoke` and {py:meth}`~aiq.builder.function.Function._astream` methods. For example, the following function has a single output type of `MySingleOutput`, and a streaming output type of `MyStreamingOutput`:

```python
class MyFunction(Function[MyInput, MySingleOutput, MyStreamingOutput]):

    async def _ainvoke(self, value: MyInput) -> MySingleOutput:
        return MySingleOutput(value)

    async def _astream(self, value: MyInput) -> AsyncGenerator[MyStreamingOutput]:
        for i in range(10):
            yield MyStreamingOutput(value, i)
```

Similarly this can be accomplished using {py:meth}`~aiq.builder.function_info.FunctionInfo.create` which is a more verbose version of {py:meth}`~aiq.builder.function_info.FunctionInfo.from_fn`.

```python
async def my_ainvoke(self, value: MyInput) -> MySingleOutput:
    return MySingleOutput(value)

async def my_astream(self, value: MyInput) -> AsyncGenerator[MyStreamingOutput]:
    for i in range(10):
        yield MyStreamingOutput(value, i)


function_info = FunctionInfo.create(
    single_fn=my_ainvoke,
    stream_fn=my_astream,
)

assert function_info.single_output_type == MySingleOutput
assert function_info.stream_output_type == MyStreamingOutput
```

Finally, when using {py:meth}`~aiq.builder.function_info.FunctionInfo.create` a conversion function can be provided to convert the single output to a streaming output, and a streaming output into a single output. This is useful when converting between streaming and single outputs is trivial and defining both methods would be overkill. For example, the following function converts a streaming output to a single output by joining the items with a comma:

```python
# Define a conversion function to convert a streaming output to a single output
def convert_streaming_to_single(value: AsyncGenerator[str]) -> str:
    return ", ".join(value)

# Define a streaming function
async def my_streaming_fn(value: str) -> AsyncGenerator[str]:
    for item in value.split(","):
        yield item

# Create a function info
function_info = FunctionInfo.create(
    single_fn=my_ainvoke,
    stream_to_single_fn=convert_streaming_to_single
)
```

### Overriding the Input and Output Schemas

It is possible to override the input and output schemas when creating a function from a callable. This is useful when it's not possible to annotate the input and output types of the callable to add validation or documentation. For example, the following function accepts a simple string and returns a string but we provide a custom input schema to add validation and documentation.

```python
@register_function(config_type=MyFunctionConfig)
async def my_function(config: MyFunctionConfig, builder: Builder):

    async def _my_function(message: str) -> str:
        # Implement the function logic
        return message

    class MyInputSchema(BaseModel):
        message: str = Field(description="This will be the message that is returned", min_length=10)

    yield FunctionInfo.from_fn(
        _my_function,
        description="My function description",
        input_schema=MyInputSchema
    )
```

When invoking the function with invalid input, the function will raise a validation error.

```python
try:
    result = await function.ainvoke("short")
except ValidationError as e:
    print(e)
```

Output schemas can also be overridden in a similar manner but for different purposes. Generally, output schemas are mainly used for adding documentation to the output of the function.

## Instantiating Functions

Once a function is registered, it can be instantiated using the {py:class}`~aiq.builder.workflow_builder.WorkflowBuilder` class. The `WorkflowBuilder` class is used to create and manage all components in an AIQ Toolkit workflow. When calling {py:meth}`~aiq.builder.workflow_builder.WorkflowBuilder.add_function`, which function to create is determined by the type of the configuration object. The builder will match the configuration object type to the type used in the {py:deco}`aiq.cli.register_workflow.register_function` decorator.

```python

class MyFunctionConfig(FunctionBaseConfig, name="my_function_id"):
    # Sample configuration options
    ...

# Register the function
@register_function(config_type=MyFunctionConfig)
async def my_function(config: MyFunctionConfig, builder: Builder):
    ...

# Create a builder
async with WorkflowBuilder() as builder:

    # Add the function to the builder. This will create an instance of my_function
    function = await builder.add_function(name="my_function", config=MyFunctionConfig())

    # Use the function directly
    result = await function.ainvoke("Hello, world!")

# The function will be automatically cleaned up when the builder is exited
```

## Invoking Functions

Functions can be invoked in two ways:

* **For single outputs**:

    ```python
    # Get a single result
    result = await function.ainvoke(input_data)
    ```

* **For streaming outputs**:

    ```python
    # Process streaming results
    async for item in function.astream(input_data):
        # Use the streaming result
        print(item)
    ```

If the function only has a single output, using the {py:meth}`~aiq.builder.function.Function.astream` method will result in an error. Likewise, if the function only has a streaming output, using the {py:meth}`~aiq.builder.function.Function.ainvoke` method will result in an error. It's possible to check which output types a function supports using the {py:attr}`~aiq.builder.function.Function.has_single_output` and {py:attr}`~aiq.builder.function.Function.has_streaming_output` properties.

## Function Composition

Functions can call other functions allowing for complex workflows to be created. To accomplish this, we can use the {py:class}`~aiq.builder.workflow_builder.WorkflowBuilder` class to get a reference to another function while constructing the current function. For example, the following function composes two other functions:

```python
class MyCompositeFunctionConfig(FunctionBaseConfig, name="my_composite_function"):
    other_function_name1: FunctionRef
    other_function_name2: FunctionRef

@register_function(config_type=MyCompositeFunctionConfig)
async def my_function(config: MyCompositeFunctionConfig, builder: Builder):

    # Get a reference to another function
    other_function1 = builder.get_function(config.other_function_name1)
    other_function2 = builder.get_function(config.other_function_name2)

    async def _my_function(message: str) -> str:

        # First call other_function1
        result1 = await other_function1.ainvoke(message)

        # Then call other_function2
        result2 = await other_function2.ainvoke(result1)

        # Return the final result
        return result2

    yield _my_function
```

:::{note}
We annotate function names in the configuration object using {py:class}`~aiq.data_models.component_ref.FunctionRef` which is equivalent to `str` but indicates that the function name is a reference to another function. When a function is referenced in a configuration object in this way, the builder system will ensure that the function is registered before it is used.
:::

## Type Conversion

When working with functions, it is not guaranteed that the input and output types will be the same as the types specified in the function definition. To make this easier, functions support type conversion which can convert both inputs and outputs to the necessary type at runtime.

To convert a value to a different type, use the {py:meth}`~aiq.builder.function.Function.convert` method where the first argument is the value to convert and the second argument, `to_type`, is the type to convert to.

```python
# Convert between types
result = function.convert(value, to_type=TargetType)
```

The {py:meth}`~aiq.builder.function.Function.convert` method is used internally by the {py:meth}`~aiq.builder.function.Function.ainvoke` and {py:meth}`~aiq.builder.function.Function.astream` methods to convert the input and output values to the necessary types. When passing a value to the {py:meth}`~aiq.builder.function.Function.ainvoke` or {py:meth}`~aiq.builder.function.Function.astream` methods, the value will be converted to the type specified by the function's input type. The {py:meth}`~aiq.builder.function.Function.ainvoke` and {py:meth}`~aiq.builder.function.Function.astream` methods effectively do the following:

```python
async def ainvoke(value: typing.Any, ...):
    # Effectively do the following
    converted_value = self.convert(value, to_type=self.input_type)

    return await self._ainvoke(converted_value)
```

Once the output is generated, the output type can be converted before it is returned using the `to_type` property on {py:meth}`~aiq.builder.function.Function.ainvoke` and {py:meth}`~aiq.builder.function.Function.astream` methods. The `to_type` property is a type hint that can be used to convert the output to a specific type using the {py:meth}`~aiq.builder.function.Function.convert` method. This is equivalent to the following:

```python
async def ainvoke(value: typing.Any, to_type: type):

    result = await self._ainvoke(value)

    return self.convert(result, to_type=to_type)
```

### Adding Custom Converters

Functions support custom type converters for complex conversion scenarios. To add a custom converter to a function, provide a list of converter callables to the {py:meth}`~aiq.builder.function_info.FunctionInfo.from_fn` or {py:meth}`~aiq.builder.function_info.FunctionInfo.create` methods when creating a function. A converter callable is any python function which takes a single value and returns a converted value. These functions must be annotated with the type it will convert from and the type it will convert to.

For example, the following converter will convert an `int` to a `str`:

```python
def my_converter(value: int) -> str:
    return str(value)
```

This converter can then be passed to the {py:meth}`~aiq.builder.function_info.FunctionInfo.from_fn` or {py:meth}`~aiq.builder.function_info.FunctionInfo.create` methods when registering the function:

```python
@register_function(config_type=MyFunctionConfig)
async def my_function(config: MyFunctionConfig, builder: Builder):

    async def _my_function(input_data: MyInput) -> AsyncGenerator[MyStreamingOutput]:
        # Implement the function logic
        for i in range(10):
            yield MyStreamingOutput(i)

    def convert_str_to_myinput(value: str) -> MyInput:
        return MyInput(value)

    yield FunctionInfo.from_fn(
        _my_function,
        description="My function description",
        converters=[convert_str_to_myinput, my_converter]
    )
```

Every function has its own set of converters and are independent of the converters used by other functions. This allows for functions to convert between common types such as `str` -> `dict` or `int` -> `float` without breaking the type safety of other functions.
