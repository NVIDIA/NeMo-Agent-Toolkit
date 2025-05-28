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

# Using Cursor Rules for Development Tasks

The AIQ toolkit integrates with Cursor to streamline common development tasks through natural language requests. Cursor rules allow you to interact with your AIQ workflows and project files using simple chat commands, which are automatically translated into the appropriate CLI commands and file edits.

This tutorial describes the tasks you can accomplish using Cursor rules and how to phrase your Cursor Chat Agent requests.

## Supported Tasks

### 1. Create a New Workflow

You can create a new workflow by simply asking:
```text
create a workflow named my_workflow
```
or specify a directory:
```text
create a workflow named my_workflow in the examples directory
```

Cursor will create the workflow for you and verify the directory structure and installation.

### 2. Delete or Uninstall a Workflow

To remove a workflow and its files:
```text
delete the workflow named my_workflow
```
or
```text
uninstall workflow my_workflow
```
Cursor will run the necessary CLI commands to uninstall and clean up the workflow.

### 3. Reinstall a Workflow

If you need to reinstall a workflow:
```text
reinstall workflow my_workflow
```
Cursor will reinstall the workflow and verify the installation.

### 4. Run a Workflow

You can run a workflow by specifying the config file and input:
```text
run workflow examples/simple/configs/config.yml with input "What is LangSmith?"
```
You can also provide override arguments:
```text
run workflow examples/simple/configs/config.yml with input "What is LangSmith?" and override llms.nim_llm.temperature to 0.7
```
Or, just specify the workflow name and input (Cursor will auto-discover the config file):
```text
run workflow simple with input "What is LangSmith?"
```

### 5. Serve a Workflow

To serve a workflow as a FastAPI endpoint:
```text
serve workflow examples/simple/configs/config.yml
```
You can also specify host and port:
```text
serve workflow examples/simple/configs/config.yml on host 0.0.0.0 port 8000
```
Additional options like workers and auto-reload can be specified:
```text
serve workflow examples/simple/configs/config.yml with 4 workers and auto-reload enabled
```

### 6. Evaluate a Workflow

To evaluate a workflow with a dataset:
```text
evaluate workflow examples/simple/configs/config.yml with dataset path/to/dataset.json
```
You can specify additional evaluation parameters:
```text
evaluate workflow examples/simple/configs/config.yml with dataset path/to/dataset.json and 3 repetitions
```

### 7. Validate a Configuration

To validate a workflow configuration file:
```text
validate config examples/simple/configs/config.yml
```
Cursor will check if the configuration file has the correct settings, components, and parameters.

### 8. Get Component Information

You can search for specific components by providing a query:
```text
Show all the tools that contains "webpage" in the name
```

or provide a description of the component you are looking for:
```text
Is there a tool that can query a webpage?
```
You can also ask for a list of all the components of a specific type:
```text
Show all the tools
```

### 9. Add a Tool to a Workflow

To add a new tool to a workflow configuration:
```text
add tool current_datetime to workflow examples/simple/configs/config.yml
```
or
```text
add webpage_query tool to workflow examples/simple/configs/config.yml with url https://docs.smith.langchain.com/user_guide
```
Cursor will update the YAML config, add the tool to the `functions` section, and update the `tool_names` list.

### 10. Customize Workflow Configuration

You can request changes to workflow parameters, add or remove tools, or update settings by describing your intent in natural language. Cursor will interpret your request and make the necessary file edits.

## How It Works

- **Natural Language Input:** You describe your intent in plain English in the Cursor chat.
- **Automatic Command Execution:** Cursor rules parse your request and run the appropriate CLI commands or file edits.
- **Validation and Feedback:** Cursor checks for errors, validates changes, and provides feedback or suggestions if something goes wrong.

## Tips

- Be as specific as possible in your requests for best results.
- You can chain multiple actions in a single request (e.g., create a workflow and add a tool).
- If a command requires confirmation (like deletion), Cursor will handle it automatically.
- For complex commands, you can specify multiple parameters in a natural way.

## Example Workflow

1. **Create a workflow:**
   ```
   create a workflow named simple_workflow in examples
   ```
2. **Add a tool:**
   ```
   add tool current_datetime to workflow examples/simple_workflow/configs/config.yml
   ```
3. **Validate the configuration:**
   ```
   validate config examples/simple_workflow/configs/config.yml
   ```
4. **Serve the workflow:**
   ```
   serve workflow examples/simple_workflow/configs/config.yml on host 0.0.0.0 port 8000
   ```
5. **Run the workflow:**
   ```
   run workflow simple_workflow with input "What is the current date and time?"
   ```
