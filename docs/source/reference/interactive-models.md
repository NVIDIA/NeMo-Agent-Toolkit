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

# Interactive Models Guide
AIQ Toolkit provides interactive prompt and response Pydantic data models as a way to validate, serialize, and document
data structures to support human input during the execution of an agent workflow.
**Note**: All human in the loop interaction data models are supported by the `aiq serve` command, while the `aiq run`
command **only** supports the {py:mod}`aiq.data_models.interactive.HumanPromptText` data model. Ensure WebSocket mode
is enabled by toggling the setting in the top-right corner of the webpage for proper interaction when using this feature
with the front-end user interface.

## How to Use Interactive Prompt and Response Data Models
Start by acquiring an instance of the {class}`aiq.builder.user_interaction_manager.AIQUserInteractionManager` class
from the {class}`aiq.builder.context.AIQContext` instance.
```python
aiq_context = AIQContext.get()
user_input_manager = aiq_context.user_interaction_manager
```

Once the {py:mod}`aiq.builder.user_interaction_manager.AIQUserInteractionManager` has been acquired, use the Interaction
Prompt data models located here: {py:mod}`aiq.data_models.interactive` to create a user defined prompt of your choosing
i.e. {py:mod}`aiq.data_models.interactive.HumanPromptText` to prompt user interaction during work flow execution.
```python
human_prompt_text = HumanPromptText(text="Hello, how are you today?", required=True, placeholder="default")
```

Pass the interaction prompt instance to the `prompt_user_input` method from the {py:mod}`aiq.builder.user_interaction_manager.AIQUserInteractionManager`  Once called the workflow will pause execution and wait for user input which can be handled
by processing the returned interaction response instance.
```python
response = await user_input_manager.prompt_user_input(human_prompt_text)
```

Finally, process the returned response from the user input.
**Note**: The response will be an instance of the corresponding data model that matches the type of user-defined interactive prompt.
```python
assert (isinstance(response.content, HumanResponseText))
return response.content.text
```

Complete example:
```python
async def _inner(prompt: str) -> str:
    try:
        aiq_context = AIQContext.get()
        user_input_manager = aiq_context.user_interaction_manager

        human_prompt_text = HumanPromptText(text="Hello, how are you today?", required=True, placeholder="default")

        response = await user_input_manager.prompt_user_input(human_prompt_text)

        assert (isinstance(response.content, HumanResponseText))

        return response.content.text

    except Exception as e:
        logger.error("An error occurred when getting interaction content: %s", e)

        raise
```
