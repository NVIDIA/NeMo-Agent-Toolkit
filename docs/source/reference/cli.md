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

# Command Line Interface

## Overview

While the NeMo Agent toolkit library provides the capability to implement components that come together to form Agentic AI
workflow, the command line interface (CLI) provides a no code entrypoint to configure settings, access the features of
pre-built components, and mechanisms to launch workflows from configuration files. This document describes the layout
and functionality of the NeMo Agent toolkit CLI. To begin, the command hierarchy is depicted below. Each command will be introduced
throughout the remainder of this document.

```
nat
├── configure
│   └── channel
│       ├── add
│       ├── remove
│       └── update
├── eval
├── info
│   ├── channels
│   └── components
├── mcp
│   ├── client
│   │   ├── ping
│   │   └── tool
│   │       ├── call
│   │       └── list
│   └── serve
├── object-store
│   ├── mysql
│   │   ├── delete
│   │   └── upload
│   ├── redis
│   │   ├── delete
│   │   └── upload
│   └── s3
│       ├── delete
│       └── upload
├── optimize
├── registry
│   ├── publish
│   ├── pull
│   ├── remove
│   └── search
├── run
├── serve
├── sizing
│   └── calc
├── start
│   ├── console
│   ├── fastapi
│   └── mcp
├── uninstall
├── validate
└── workflow
    ├── create
    ├── delete
    └── reinstall
```

## Start

The group of `nat start` commands provide several mechanisms to launch workflows. Each of these commands are summarized
in the following sections.

### FastAPI

The `nat start fastapi` command will serve a FastAPI endpoint for the workflow based on the supplied configuration file
in the `--config_file` option. This command is ideal for serving a workflow as a microservice that allows client
applications to submit requests to a workflow. The `nat serve` command is a good option when deploying this workflow into
production as the entrypoint of a containerized application. Additional options are available to serve this workflow
are made available via the `nat start fastapi --help` utility:

```console
$ nat start fastapi --help
Usage: nat start fastapi [OPTIONS]

Options:
  --config_file FILE              A JSON/YAML file that sets the parameters
                                  for the workflow.  [required]
  --override <TEXT TEXT>...       Override config values using dot notation
                                  (e.g., --override llms.nim_llm.temperature
                                  0.7)
  --root_path TEXT                The root path for the API
  --host TEXT                     Host to bind the server to
  --port INTEGER                  Port to bind the server to
  --reload BOOLEAN                Enable auto-reload for development
  --workers INTEGER               Number of workers to run
  --step_adaptor STEPADAPTORCONFIG
  --workflow ENDPOINTBASE         Endpoint for the default workflow.
  --endpoints ENDPOINT            Additional endpoints to add to the FastAPI
                                  app which run functions within the NAT
                                  configuration. Each endpoint must have a
                                  unique path.
  --use_gunicorn BOOLEAN          Use Gunicorn to run the FastAPI app
  --runner_class TEXT             The NAT runner class to use when launching
                                  the FastAPI app from multiple processes.
                                  Each runner is responsible for loading and
                                  running the NAT workflow. Note: This is
                                  different from the worker class used by
                                  Gunicorn.
  --help                          Show this message and exit.
```

Once a workflow has been launched using the `nat start fastapi` command, client applications may submit POST requests
that will run data through the hosted workflow. To access documentation on the available routes and schemas, Swagger API
documentation are made available at the <HOST>:<IP>/docs endpoint. For example, if serving locally, with
the following command:

<!-- path-check-skip-begin -->
```bash
nat start fastapi --config_file=path/to/config --host 0.0.0.0 --port 8000
```
<!-- path-check-skip-end -->

The Swagger API docs will be available at: [http://localhost:8000/docs](http://localhost:8000/docs)

### Console

The `nat start console` command will run a NeMo Agent toolkit workflow from a provided configuration file against inputs supplied
at the command line or from file using the `--inputs` and `--input_file` options, respectively. Additionally, fields in
the configuration file can be overridden by command line using the `--override` flag and dot notation to traverse to the
configuration hierarchy to the field being overridden. The run command can be useful running one off tests when
debugging a workflow. When invoking the run command, the workflow will follow the same harness as the
other workflow launch commands. This simplifies the debugging process when transitioning from development to production.

The `nat start console` help utility provides a brief description of each option to describe is usage.

```console
$ nat start console --help
Usage: nat start console [OPTIONS]

Options:
  --config_file FILE         A JSON/YAML file that sets the parameters for the
                             workflow.  [required]
  --override <TEXT TEXT>...  Override config values using dot notation (e.g.,
                             --override llms.nim_llm.temperature 0.7)
  --input TEXT               A single input to submit the the workflow.
  --input_file FILE          Path to a json file of inputs to submit to the
                             workflow.
  --help                     Show this message and exit.
```

## MCP

The `nat mcp` command group provides utilities for both serving workflows as MCP servers and interacting with MCP servers as a client.

### Client

The `nat mcp client` command group provides utilities for interacting with MCP servers directly from the command line. These commands are useful for discovering available tools and testing MCP server connectivity before configuring your workflow.

The `nat mcp client --help` utility provides an overview of the available commands:

```console
$ nat mcp client --help
Usage: nat mcp client [OPTIONS] COMMAND [ARGS]...

  MCP client commands.

Options:
  --help  Show this message and exit.

Commands:
  ping  Ping an MCP server to check if it's responsive.
  tool  Inspect and call MCP tools.
```

#### Ping

```console
$ nat mcp client ping --help
Usage: nat mcp client ping [OPTIONS]

  Ping an MCP server to check if it's responsive.

Options:
  --url TEXT                      MCP server URL (e.g.
                                  http://localhost:8080/mcp for streamable-
                                  http, http://localhost:8080/sse for sse)
                                  [default: http://localhost:9901/mcp]
  --transport [sse|stdio|streamable-http]
                                  Type of client to use for ping  [default:
                                  streamable-http]
  --command TEXT                  For stdio: The command to run (e.g. mcp-
                                  server)
  --args TEXT                     For stdio: Additional arguments for the
                                  command (space-separated)
  --env TEXT                      For stdio: Environment variables in
                                  KEY=VALUE format (space-separated)
  --timeout INTEGER               Timeout in seconds for ping request
                                  [default: 60]
  --json-output                   Output ping result in JSON format
  --auth-redirect-uri TEXT        OAuth2 redirect URI for authentication
                                  (streamable-http only, not with --direct)
  --auth-user-id TEXT             User ID for authentication (streamable-http
                                  only, not with --direct)
  --auth-scopes TEXT              OAuth2 scopes (comma-separated, streamable-
                                  http only, not with --direct)
  --help                          Show this message and exit.
```

#### Tool Commands

```console
$ nat mcp client tool --help
Usage: nat mcp client tool [OPTIONS] COMMAND [ARGS]...

  Inspect and call MCP tools.

Options:
  --help  Show this message and exit.

Commands:
  call  Call a tool by name with optional arguments.
  list  List tool names (default), or show details with --detail or --tool.
```

##### List Tools

```console
$ nat mcp client tool list --help
Usage: nat mcp client tool list [OPTIONS]

  List tool names (default), or show details with --detail or --tool.

Options:
  --direct                        Bypass MCPBuilder and use direct MCP
                                  protocol
  --url TEXT                      MCP server URL (e.g.
                                  http://localhost:8080/mcp for streamable-
                                  http, http://localhost:8080/sse for sse)
                                  [default: http://localhost:9901/mcp]
  --transport [sse|stdio|streamable-http]
                                  Type of client to use (default: streamable-
                                  http, backwards compatible with sse)
                                  [default: streamable-http]
  --command TEXT                  For stdio: The command to run (e.g. mcp-
                                  server)
  --args TEXT                     For stdio: Additional arguments for the
                                  command (space-separated)
  --env TEXT                      For stdio: Environment variables in
                                  KEY=VALUE format (space-separated)
  --tool TEXT                     Get details for a specific tool by name
  --detail                        Show full details for all tools
  --json-output                   Output tool metadata in JSON format
  --auth                          Enable OAuth2 authentication with default
                                  settings (streamable-http only, not with
                                  --direct)
  --auth-redirect-uri TEXT        OAuth2 redirect URI for authentication
                                  (streamable-http only, not with --direct)
  --auth-user-id TEXT             User ID for authentication (streamable-http
                                  only, not with --direct)
  --auth-scopes TEXT              OAuth2 scopes (comma-separated, streamable-
                                  http only, not with --direct)
  --help                          Show this message and exit.
```

##### Call Tool

```console
$ nat mcp client tool call --help
Usage: nat mcp client tool call [OPTIONS] TOOL_NAME

  Call a tool by name with optional arguments.

Options:
  --direct                        Bypass MCPBuilder and use direct MCP
                                  protocol
  --url TEXT                      MCP server URL (e.g.
                                  http://localhost:8080/mcp for streamable-
                                  http, http://localhost:8080/sse for sse)
                                  [default: http://localhost:9901/mcp]
  --transport [sse|stdio|streamable-http]
                                  Type of client to use (default: streamable-
                                  http, backwards compatible with sse)
                                  [default: streamable-http]
  --command TEXT                  For stdio: The command to run (e.g. mcp-
                                  server)
  --args TEXT                     For stdio: Additional arguments for the
                                  command (space-separated)
  --env TEXT                      For stdio: Environment variables in
                                  KEY=VALUE format (space-separated)
  --json-args TEXT                Pass tool args as a JSON object string
  --auth                          Enable OAuth2 authentication with default
                                  settings (streamable-http only, not with
                                  --direct)
  --auth-redirect-uri TEXT        OAuth2 redirect URI for authentication
                                  (streamable-http only, not with --direct)
  --auth-user-id TEXT             User ID for authentication (streamable-http
                                  only, not with --direct)
  --auth-scopes TEXT              OAuth2 scopes (comma-separated, streamable-
                                  http only, not with --direct)
  --help                          Show this message and exit.
```

### Serve

The `nat mcp serve` command (equivalent to `nat start mcp`) starts a Model Context Protocol (MCP) server that exposes workflow functions as MCP tools. This allows other applications that support the MCP protocol to use your NeMo Agent toolkit functions directly. MCP is an open protocol developed by Anthropic that standardizes how applications provide context to LLMs.

The `nat mcp serve --help` utility provides a brief description of each option:

```console
$ nat mcp serve --help
Usage: nat mcp serve [OPTIONS]

  Run a NAT workflow using the mcp front end.

Options:
  --config_file FILE              A JSON/YAML file that sets the parameters
                                  for the workflow.  [required]
  --override <TEXT TEXT>...       Override config values using dot notation
                                  (e.g., --override llms.nim_llm.temperature
                                  0.7)
  --name TEXT                     Name of the MCP server (default: NeMo Agent
                                  Toolkit MCP)
  --host TEXT                     Host to bind the server to (default:
                                  localhost)
  --port INTEGER                  Port to bind the server to (default: 9901)
  --debug BOOLEAN                 Enable debug mode (default: False)
  --log_level TEXT                Log level for the MCP server (default: INFO)
  --tool_names TEXT               The list of tools MCP server will expose
                                  (default: all tools)
  --transport [sse|streamable-http]
                                  Transport type for the MCP server (default:
                                  streamable-http, backwards compatible with
                                  sse)
  --runner_class TEXT             Custom worker class for handling MCP routes
                                  (default: built-in worker)
  --server_auth OAUTH2RESOURCESERVERCONFIG
                                  OAuth 2.0 Resource Server configuration for
                                  token verification.
  --help                          Show this message and exit.
```

For example, to start an MCP server with a specific workflow:

```bash
nat mcp serve --config_file examples/getting_started/simple_calculator/configs/config.yml
```

This will start an MCP server on the default host (localhost) and port (9901), available at `http://localhost:9901/mcp`.

## Run

The `nat run` is an alias for the `nat start console` command and will run a NeMo Agent toolkit workflow from a provided configuration file against inputs supplied at the
command line or from file using the `--inputs` and `--input_file` options, respectively. Additionally, fields in the
configuration file can be overridden by command line using the `--override` flag and dot notation to traverse to the
configuration hierarchy to the field being overridden. The run command can be useful running one off tests when
debugging a workflow. When invoking the run command, the workflow will follow the same harness as the
other workflow launch commands. This simplifies the debugging process when transitioning from development to production.

The `nat run` help utility provides a brief description of each option to describe is usage.

```console
$ nat run --help
Usage: nat run [OPTIONS]

Options:
  --config_file FILE         A JSON/YAML file that sets the parameters for the
                             workflow.  [required]
  --override <TEXT TEXT>...  Override config values using dot notation (e.g.,
                             --override llms.nim_llm.temperature 0.7)
  --input TEXT               A single input to submit the the workflow.
  --input_file FILE          Path to a json file of inputs to submit to the
                             workflow.
  --help                     Show this message and exit.
```

## Serve
The `nat serve` is an alias for the `nat start fastapi` command and will serve a FastAPI endpoint for the workflow based
on the supplied configuration file in the `--config_file` option. This command is ideal for serving a workflow as a
microservice that allows client applications to submit requests to a workflow. The `nat serve` command is a good option
when deploying this workflow into production as the entrypoint of a containerized application. Additional options are
available to serve this workflow are made available via the `nat serve --help` utility:

```console
$ nat serve --help
Usage: nat serve [OPTIONS]

Options:
  --config_file FILE              A JSON/YAML file that sets the parameters
                                  for the workflow.  [required]
  --override <TEXT TEXT>...       Override config values using dot notation
                                  (e.g., --override llms.nim_llm.temperature
                                  0.7)
  --root_path TEXT                The root path for the API
  --host TEXT                     Host to bind the server to
  --port INTEGER                  Port to bind the server to
  --reload BOOLEAN                Enable auto-reload for development
  --workers INTEGER               Number of workers to run
  --step_adaptor STEPADAPTORCONFIG
  --workflow ENDPOINTBASE         Endpoint for the default workflow.
  --endpoints ENDPOINT            Additional endpoints to add to the FastAPI
                                  app which run functions within the NAT
                                  configuration. Each endpoint must have a
                                  unique path.
  --use_gunicorn BOOLEAN          Use Gunicorn to run the FastAPI app
  --runner_class TEXT             The NAT runner class to use when launching
                                  the FastAPI app from multiple processes.
                                  Each runner is responsible for loading and
                                  running the NAT workflow. Note: This is
                                  different from the worker class used by
                                  Gunicorn.
  --help                          Show this message and exit.
```

Once a workflow has been launched using the `nat serve` command, client applications may submit POST requests that will
run data through the hosted workflow. To access documentation on the available routes and schemas, Swagger API
documentation are made available at the <HOST>:<IP>/docs endpoint. For example, if serving locally, with
the following command:

<!-- path-check-skip-begin -->
```bash
nat serve --config_file=path/to/config --host 0.0.0.0 --port 8000
```
<!-- path-check-skip-end -->

The Swagger API docs will be available at: [http://localhost:8000/docs](http://localhost:8000/docs)

## Evaluation
The `nat eval` command provides access a set of evaluators designed to assessing the accuracy of NeMo Agent toolkit workflows as
well as instrumenting their performance characteristics. Please reference
[Evaluating NeMo Agent toolkit Workflows](../workflows/evaluate.md) for a detailed overview of the
suite of evaluation capabilities.

The `nat eval --help` utility provides a brief overview of the command and its available options.

```console
$ nat eval --help
Usage: nat eval [OPTIONS] COMMAND [ARGS]...

  Evaluate a workflow with the specified dataset.

Options:
  --config_file FILE          A JSON/YAML file that sets the parameters for
                              the workflow and evaluation.  [required]
  --dataset FILE              A json file with questions and ground truth
                              answers. This will override the dataset path in
                              the config file.
  --result_json_path TEXT     A JSON path to extract the result from the
                              workflow. Use this when the workflow returns
                              multiple objects or a dictionary. For example,
                              '$.output' will extract the 'output' field from
                              the result.  [default: $]
  --skip_workflow             Skip the workflow execution and use the provided
                              dataset for evaluation. In this case the dataset
                              should have the 'generated_' columns.
  --skip_completed_entries    Skip the dataset entries that have a generated
                              answer.
  --endpoint TEXT             Use endpoint for running the workflow. Example:
                              http://localhost:8000/generate
  --endpoint_timeout INTEGER  HTTP response timeout in seconds. Only relevant
                              if endpoint is specified.  [default: 300]
  --reps INTEGER              Number of repetitions for the evaluation.
                              [default: 1]
  --help                      Show this message and exit.
```

## Optimize

The `nat optimize` command provides automated hyperparameter tuning and prompt engineering for NeMo Agent toolkit workflows. It intelligently searches for the best combination of parameters based on the evaluation metrics you specify. The optimizer uses [Optuna](https://optuna.org/) for numerical hyperparameter optimization and a genetic algorithm (GA) for prompt optimization. Please reference the [NeMo Agent toolkit Optimizer Guide](../reference/optimizer.md) for a comprehensive overview of the optimizer capabilities and configuration.

The `nat optimize --help` utility provides a brief overview of the command and its available options:

```console
$ nat optimize --help
Usage: nat optimize [OPTIONS] COMMAND [ARGS]...

  Optimize a workflow with the specified dataset.

Options:
  --config_file FILE          A JSON/YAML file that sets the parameters for
                              the workflow and evaluation.  [required]
  --dataset FILE              A json file with questions and ground truth
                              answers. This will override the dataset path in
                              the config file.
  --result_json_path TEXT     A JSON path to extract the result from the
                              workflow. Use this when the workflow returns
                              multiple objects or a dictionary. For example,
                              '$.output' will extract the 'output' field from
                              the result.  [default: $]
  --endpoint TEXT             Use endpoint for running the workflow. Example:
                              http://localhost:8000/generate
  --endpoint_timeout INTEGER  HTTP response timeout in seconds. Only relevant
                              if endpoint is specified.  [default: 300]
  --help                      Show this message and exit.
```

### Options Description

- **`--config_file`**: This is the main configuration file that contains both the workflow configuration and the optimizer settings. The file must include an `optimizer` section that defines the optimization parameters, search spaces, and evaluation metrics.

- **`--dataset`**: Path to a JSON file containing the evaluation dataset with questions and ground truth answers. If provided, this will override the dataset path specified in the configuration file. The dataset is used to evaluate different parameter combinations during optimization.

- **`--result_json_path`**: A JSON path expression to extract the relevant result from the workflow output. This is useful when your workflow returns complex objects or dictionaries and you need to specify which field contains the actual result to evaluate. The default value `$` uses the entire output.

- **`--endpoint`**: Instead of running the workflow locally, you can specify an HTTP endpoint where the workflow is deployed. This is useful for optimizing workflows that are already running as services.

- **`--endpoint_timeout`**: When using the `--endpoint` option, this sets the maximum time (in seconds) to wait for a response from the remote service.


To optimize a workflow with a local configuration, run:

<!-- path-check-skip-begin -->
```bash
nat optimize --config_file configs/my_workflow_optimizer.yml
```
<!-- path-check-skip-end -->

## GPU Cluster Sizing

The `nat sizing calc` command estimates GPU requirements and produces performance plots for a workflow. You can run it online (collect metrics by executing the workflow) or offline (estimate from previously collected metrics). For a full guide, see [GPU Cluster Sizing](../workflows/sizing-calc.md).

The `nat sizing calc --help` utility provides a brief overview of the command and its available options:

```console
$ nat sizing calc --help
Usage: nat sizing calc [OPTIONS]

  Estimate GPU count and plot metrics for a workflow

Options:
  --config_file FILE               A YAML config file for the workflow and
                                   evaluation. This is not needed in offline
                                   mode.
  --offline_mode                   Run in offline mode. This is used to
                                   estimate the GPU count for a workflow
                                   without running the workflow.
  --target_llm_latency FLOAT       Target p95 LLM latency (seconds). Can be
                                   set to 0 to ignore.
  --target_workflow_runtime FLOAT  Target p95 workflow runtime (seconds). Can
                                   be set to 0 to ignore.
  --target_users INTEGER           Target number of users to support.
  --test_gpu_count INTEGER         Number of GPUs used in the test.
  --calc_output_dir DIRECTORY      Directory to save plots and results
                                   (optional).
  --concurrencies TEXT             Comma-separated list of concurrency values
                                   to test (e.g., 1,2,4,8). Default:
                                   1,2,3,4,5,6,7,8,9,10
  --num_passes INTEGER             Number of passes at each concurrency for the
                                   evaluation. If set to 0 the dataset is
                                   adjusted to a multiple of the concurrency.
                                   Default: 0
  --append_calc_outputs            Append calc outputs to the output
                                   directory. By default append is set to
                                   False and the content of the online
                                   directory is overwritten.
  --endpoint TEXT                  Endpoint to use for the workflow if it is
                                   remote (optional).
  --endpoint_timeout INTEGER       Timeout for the remote workflow endpoint in
                                   seconds (default: 300).
  --help                           Show this message and exit.
```

### Examples

- Online metrics collection and plots:

```bash
nat sizing calc \
  --config_file $CONFIG_FILE \
  --calc_output_dir $CALC_OUTPUT_DIR \
  --concurrencies 1,2,4,8,16,32 \
  --num_passes 2
```

- Offline estimation from prior results, targeting 100 users and 10-second p95 workflow time, assuming tests ran with 8 GPUs:

```bash
nat sizing calc \
  --offline_mode \
  --calc_output_dir $CALC_OUTPUT_DIR \
  --test_gpu_count 8 \
  --target_workflow_runtime 10 \
  --target_users 100
```

- Combined run (collect metrics and estimate in one command):

```bash
nat sizing calc \
  --config_file $CONFIG_FILE \
  --calc_output_dir $CALC_OUTPUT_DIR \
  --concurrencies 1,2,4,8,16,32 \
  --num_passes 2 \
  --test_gpu_count 8 \
  --target_workflow_runtime 10 \
  --target_users 100
```

## Uninstall

When a package and its corresponding components are no longer needed, they can be removed from the local environment.
This can help if certain packages are creating dependency conflicts. To remove packages from the local environment, use
the `nat uninstall` command. This command can be used with one or more packages. The `nat uninstall --help` utility
illustrates is usage:

```console
$ nat uninstall --help
Usage: nat uninstall [OPTIONS] PACKAGES COMMAND [ARGS]...

  Uninstall plugin packages from the local environment.

Options:
  --help  Show this message and exit.
```

## Validate

Running a NeMo Agent toolkit workflow from the CLI requires a valid workflow configuration file. Use the `nat validate` command to
ensure a configuration files has been created with the right settings, components and parameters. It can be useful to
each components valid configuration settings using the `nat info components` command and corresponding filters.
The `nat validate` help utility illustrates its usage.

```console
$ nat validate --help
Usage: nat validate [OPTIONS]

  Validate a configuration file

Options:
  --config_file FILE  Configuration file to validate  [required]
  --help              Show this message and exit.
```

## Workflow

The extensibility of NeMo Agent toolkit is made possible through its plugin system. To install these plugins, they must be part of
a Python package that gets installed in an environment where the NeMo Agent toolkit library is installed. Creating boiler plate
package files (e.g. `pyproject.toml`) and component code scaffolding can be tedious. This section provides an overview
of commands that automate some of these steps.

### Create

The `nat workflow create` command generates a valid `pyproject.toml` file with a plugin section that points to a
register.py file that has been pre-populated with NeMo Agent toolkit programming model boiler plate code. This boiler plate code
should be further customized to implement the desired custom workflow and necessary NeMo Agent toolkit components. The
`nat workflow create --help` utility provides a description of its usage.

```console
$ nat workflow create --help
Usage: nat workflow create [OPTIONS] WORKFLOW_NAME

  Create a new NAT workflow using templates.

  Args:     workflow_name (str): The name of the new workflow.     install
  (bool): Whether to install the workflow package immediately.
  workflow_dir (str): The directory to create the workflow package.
  description (str): Description to pre-popluate the workflow docstring.

Options:
  --install / --no-install  Whether to install the workflow package
                            immediately.  [default: install]
  --workflow-dir TEXT       Output directory for saving the created workflow.
                            A new folder with the workflow name will be
                            created within.Defaults to the present working
                            directory.  [default: .]
  --description TEXT        A description of the component being created. Will
                            be used to populate the docstring and will
                            describe the component when inspecting installed
                            components using 'nat info component'  [default:
                            NAT function template. Please update the
                            description.]
  --help                    Show this message and exit.
```

Also, a configuration file will be generated when you run the `nat workflow create` command. To launch the new workflow from the CLI
(e.g. using `nat run` or `nat serve`), you will need a configuration file that maps to these component
configuration objects. For more information on configuration objects, refer to
[Workflow Configuration](../workflows/workflow-configuration.md).

### Reinstall

When you modify a workflow's code or update its dependencies, you need to reinstall the workflow package to ensure the changes take effect. The `nat workflow reinstall` command rebuilds and reinstalls the workflow package with any updates. This is particularly useful after:

- Modifying the workflow's Python code
- Updating dependencies in `pyproject.toml`
- Making changes to the workflow's configuration
- Adding new tools or components

The `nat workflow reinstall --help` utility provides a description of its usage:

```console
$ nat workflow reinstall --help
Usage: nat workflow reinstall [OPTIONS] WORKFLOW_NAME

  Reinstall a NAT workflow package.

  Args:
      workflow_name (str): The name of the workflow to reinstall.

Options:
  --help  Show this message and exit.
```

For example, after updating the dependencies in your workflow's `pyproject.toml`, you would run:

```bash
nat workflow reinstall my_workflow
```

After running the `nat workflow reinstall` command, the following actions will happen:
1. Rebuild the workflow package
2. Uninstall the existing version
3. Install the updated version
4. Verify the installation by checking the registered components

:::{note}
If you want to completely remove a workflow instead of reinstalling it, use the `nat workflow delete` command.
:::

### Delete

By default, unless the `--no-install` flag is set, the `nat workflow create` command will install the generated package
into the local environment. To remove a workflow package from the local environment, use the `nat workflow delete` command.

```console
$ nat workflow delete --help
Usage: nat workflow delete [OPTIONS] WORKFLOW_NAME

  Delete a NAT workflow and uninstall its package.

  Args:     workflow_name (str): The name of the workflow to delete.

Options:
  --help  Show this message and exit.
```


## Information Commands

The `nat info` command group provides utilities that facilitate the discovery of registered NeMo Agent toolkit components and
retrieval of information about the locally configured NeMo Agent toolkit environment.

### Components Information

When defining a NeMo Agent toolkit workflow's configuration file, it can be helpful to discover the locally registered components,
possible configuration settings, and their default values. The `nat info components` will provide this information in
tabular format with the following columns.

- `package`: The Python package containing this row's NAT component.
- `version`: The version of the Python package containing the NAT component.
- `component_type`: The type of NAT component this row represents
(e.g. `front_end`, `function`, `tool_wrapper`, `llm_provider`, `llm_client`, `embedder_provider`, `embedder_client`,
`evaluator`, `memory`, `retriever_provider`, `retriever_client`, `registry_handler`, `package`).
- `component_name`: The name of the NAT component to be specified in the `_type` field of the component's section
of the configuration file.
- `description`: A description of the component's uses, configuration parameters, and any default values. These
parameters are what will need to be specified in the configuration object.

The `nat info components --help` utility provides an overview of usage and filter options:

```console
$ nat info components --help
Usage: nat info components [OPTIONS] COMMAND [ARGS]...

  List the locally registered NAT components.

Options:
  -t, --types [front_end|function|tool_wrapper|llm_provider|llm_client|embedder_provider|embedder_client|evaluator|memory|retriever_provider|retriever_client|registry_handler|logging|tracing|package|undefined]
                                  Filter the search by NAT component type.
  -o, --output_path TEXT          Path to save search results.
  -q, --query TEXT                The query string.  [default: ""]
  -n, --num_results INTEGER       Number of results to return.  [default: -1]
  -f, --fields [all|package|version|component_name|description|developer_notes]
                                  Fields used when applying query.
  --help                          Show this message and exit.
```

### Channels Information

The `nat info channels` command provides a list of each configured remote registry channel and their corresponding
configuration settings. This command provides the `-t, --type` option to filter the remote registry channels by type.
By default, this command will return an empty list. The `nat registry` command group will not be functional without
first configuring registry channels with the `nat configure channel add` command. Successful channel configurations
will be returned when invoking the `nat info channels` command.

The `nat info channels --help` provides an overview of its usage:

```console
$ nat info channels --help
Usage: nat info channels [OPTIONS] COMMAND [ARGS]...

  List the configured remote registry channels.

Options:
  -t, --type TEXT  Filter the results by channel type.
  --help           Show this message and exit.
```

## Configuration Commands

A NeMo Agent toolkit developer may want to configure persistent settings for their development environment. These settings would be configured once to setup their development environment so they can focus on software development from that point
forward. This section discusses the various configuration settings available for NeMo Agent toolkit developers.

### Remote Registry Configuration

One of the core value propositions of the NeMo Agent toolkit library is the redistribution of components with other developers.
Being able to package and distribute packages such that other developers can leverage them is critical to accelerating
developer velocity. Similarly, being able to discover and install components built by others will improve the
current developer's velocity. To facilitate this process, NeMo Agent toolkit implements a remote registry `channel` concept that
allows NeMo Agent toolkit developers to subscribe to registries that store published NeMo Agent toolkit packages, each container containing
usable components. A `channel` is analogous to a Conda channel for Anaconda users or a PyPI registry for pip users.


#### Adding a Remote Registry Channel
Currently, there are two channel types that facilitate remote discovery and reuse:
 - `rest` – provides a contract driven interface to a registry service behind a REST endpoint
 - `pypi` – a simple interface to publish packages to a private PyPI registry.

Invoking the `nat info components` command provides a description of the available channel settings.

Here we provide a example that configures a remote rest channel. To use this channel, there must exists a remote
registry that adheres to the contracts defined in the rest handler in NeMo Agent toolkit.

```console
$ nat configure channel add rest
Channel Name: my_rest_channel  # A user defined locally unique name used to reference this configured channel
Endpoint: http://my_rest_channel_url.com  # The endpoint to the remote rest registry service
Token: my_rest_token  # The authentication token to interact with this rest registry service
Publish Route: publish  # The route to use when publishing NAT packages
Pull Route: pull  # The route to use when downloading NAT packages
Search Route: search  # The route use when searching for relevant NAT packages
Remove Route: remove  # The route to use when removing a published package from a remote rest registy
```

Here we provide a example that configures a remote `pypi` channel. This assumes there exists a private PyPI registry.

```console
$ nat configure channel add pypi
Channel Name: my_pypi_channel  # A user defined locally unique name used to reference this configured channel
Endpoint: http://my_pypi_channel_url.com  # The endpoint to the private pypi registry service
Token: my_pypi_token  # The authentication token to interact with this pypi registry service
Publish Route:  # The route to use when publishing NAT packages, setting an empty value here
Pull Route: # The route to use when downloading NAT packages, setting an empty value here
Search Route: simple  # The route use when searching for relevant NAT packages
```

#### Updating a Remote Registry Channel Configuration

At some point, a developer might need to update a remote registry channel's configuration settings. In this case,
using the `nat configure channel update` command will select a remote registry channel by its locally unique name and allow
the developer to override the configuration settings.

A usage example is provided below:

```console
$ nat configure channel update my_rest_channel
Endpoint: http://my_updated_rest_channel_url.com  # The overridden endpoint to the remote rest registry service
Token: my_rest_token
Publish Route: publish
Pull Route: pull
Search Route: search
Remove Route: remove
```

#### Removing a Remote Registry Channel

A developer may need to remove a locally configured remote registry channel. In this case, the `nat registry remove`
command can be used. The channel will be removed based on the name supplied with the command.

An example of using this command is provided below:

```bash
nat configure channel remove my_rest_channel
```

Note, once a channel is removed, it will no longer be able to support `nat registry publish`, `nat registry search`,
`nat registry pull`, or `nat registry remove` commands until reconfigured.

## Remote Registry Interactions

NeMo Agent toolkit is designed to be a community oriented library. This means that developer productivity is maximized when others
distribute NeMo Agent toolkit plugin packages that will benefit others. This section will introduce the mechanisms the NeMo Agent toolkit CLI
exposes to facilitate publishing, discovering, downloading, and removing NeMo Agent toolkit packages from a configured remote
registry. Here we define a remote registry as a centralized location that stores plugin wheel packages and NeMo Agent toolkit
specific metadata to that describes its usage details. Before these commands can be used, a remote registry must be
available and a developer must have configured the corresponding channel using the `nat configure channel add` command.
Refer to [Adding a Remote Registry Channel](#adding-a-remote-registry-channel) for more details on adding a remote registry channels.

The `nat registry` help command will provide the available commands in this group.

```console
$ nat registry --help
Usage: nat registry [OPTIONS] COMMAND [ARGS]...

  Utility to configure NAT remote registry channels.

Options:
  --help  Show this message and exit.

Commands:
  publish  Publish local NAT artifacts to a remote registry from package...
  pull     Pull NAT artifacts from a remote registry by package name.
  remove   Remove NAT artifact from a remote registry by name and version.
  search   Search for NAT artifacts from remote registry.
```

#### Publishing NeMo Agent Toolkit Components

NeMo Agent toolkit developers may want to distribute their components with the broader ecosystem. The NeMo Agent toolkit publish CLI utility
provides a mechanism to publish a NeMo Agent toolkit plugin package to a remote registry channel so that other developers can
benefit from it's implemented components. Invoking the `nat registry publish` command will build a package wheel, gather
all component metadata, and transmit to the specified remote registry by channel name. Note, a package must be first
installed locally so the discovery hooks can pull in necessary NeMo Agent toolkit component metadata.

The `nat registry publish --help` utility provides an overview of its usage:

```console
$ nat registry publish --help
Usage: nat registry publish [OPTIONS] PACKAGE_ROOT COMMAND [ARGS]...

  Publish local NAT artifacts to a remote registry from package
  repository.

Options:
  --config_file FILE  A YAML file to override configured channel settings.
  -c, --channel TEXT  The remote registry channel to use when publishing the
                      NAT artifact.  [required]
  --help              Show this message and exit.

```

#### Discovering NeMo Agent Toolkit Components

When developing and deploying NeMo Agent toolkit workflows, it is most efficient to leverage pre-built components. When using
pre-built components will, only configuration settings are required to integration with the rest of a workflow. These
pre-built exist in the core library, as well as, within other NeMo Agent toolkit plugin packages. Remote registry channels are the
formal mechanism to publish reusable components to the community. The `nat registry search` command allows developers
to search relevant pre-built components that might benefit their application. The search command is usually followed up
by an `nat registry pull` command, once a useful package has been identified.

The `nat registry search --help` utility provides an overview of its usage:

```console
$ nat registry search --help
Usage: nat registry search [OPTIONS] COMMAND [ARGS]...

  Search for NAT artifacts from remote registry.

Options:
  --config_file FILE              A JSON/YAML file that sets the parameters
                                  for the workflow.
  -c, --channel TEXT              The remote registry channel to use when
                                  pulling the NAT artifact.  [required]
  -o, --output_path TEXT          Path to save search results.
  -f, --fields [all|package|version|component_name|description|developer_notes]
                                  The fields to include in the search.
  -q, --query TEXT                The query string.  [required]
  -n, --n_results INTEGER         Number of search results to return.
                                  [default: 10]
  -t, --types [front_end|function|tool_wrapper|llm_provider|llm_client|embedder_provider|embedder_client|evaluator|memory|retriever_provider|retriever_client|registry_handler|logging|tracing|package|undefined]
                                  The component types to include in search.
  --help                          Show this message and exit.
```

#### Pulling in NeMo Agent Toolkit Components
Once a useful NeMo Agent toolkit component has been discovered using the `nat registry search` command, the containing package can be
pulled in and installed from a configured remote registry, so that it can be used withing the local NeMo Agent toolkit environment.
Once installed, all components in the package can be referenced by name in a NeMo Agent toolkit workflow YAML configuration file.
In many cases, components can be stitched together in YAML without having to write much integration code.

The `nat registry pull --help` command provides an overview of its usage:

```console
$ nat registry pull --help
Usage: nat registry pull [OPTIONS] PACKAGES COMMAND [ARGS]...

  Pull NAT artifacts from a remote registry by package name.

Options:
  --config_file FILE  A YAML file to override the channel settings.
  -c, --channel TEXT  The remote registry channel to use when pulling the
                      NAT artifact.  [required]
  --help              Show this message and exit.
```

Note, the supplied package takes the following format: `package_name==version`, where the package version is optional.


#### Removing NeMo Agent Toolkit Components
In rare cases, it might make sense to remove a package from a remote registry over a configured remote registry channel.
This the `nat registry remove` command provides support for this feature, assuming the remote registry provides and
allows this interaction.

The `nat registry remove --help` utility provides an overview of its usage.

```console
$ nat registry remove --help
Usage: nat registry remove [OPTIONS] PACKAGES COMMAND [ARGS]...

  Remove NAT artifact from a remote registry by name and version.

Options:
  --config_file FILE  A YAML file to override the channel settings.
  -c, --channel TEXT  The remote registry channel that will remove the NAT
                      artifact.  [required]
  --help              Show this message and exit.
```

## Object Store Commands

The `nat object-store` command group provides utilities to interact with object stores. This command group is used to
upload and download files to and from object stores.

The `nat object-store --help` utility provides an overview of its usage:

```console
$ nat object-store --help
Usage: nat object-store [OPTIONS] COMMAND [ARGS]...

  Manage object store operations.

Options:
  --help  Show this message and exit.

Commands:
  mysql  MySQL object store operations.
  redis  Redis object store operations.
  s3     S3 object store operations.
```

The listed commands are dependent on the first-party object store plugins installed. See [Object Store](../store-and-retrieve/object-store.md) for more details.

### MySQL Object Store

The `nat object-store mysql` command provides operations to interact with a MySQL object store.

The `nat object-store mysql --help` utility provides an overview of its usage:

```console
Usage: nat object-store mysql [OPTIONS] BUCKET_NAME COMMAND [ARGS]...

  MySQL object store operations.

Options:
  --host TEXT      MySQL host
  --port INTEGER   MySQL port
  --db TEXT        MySQL database name
  --username TEXT  MySQL username
  --password TEXT  MySQL password
  --help           Show this message and exit.

Commands:
  delete  Delete files from an object store.
  upload  Upload a directory to an object store.
```

### Redis Object Store

The `nat object-store redis` command provides operations to interact with a Redis object store.

The `nat object-store redis --help` utility provides an overview of its usage:

```console
Usage: nat object-store redis [OPTIONS] BUCKET_NAME COMMAND [ARGS]...

  Redis object store operations.

Options:
  --host TEXT     Redis host
  --port INTEGER  Redis port
  --db INTEGER    Redis db
  --help          Show this message and exit.

Commands:
  delete  Delete files from an object store.
  upload  Upload a directory to an object store.
```

### S3 Object Store

The `nat object-store s3` command provides operations to interact with a S3 object store.

The `nat object-store s3 --help` utility provides an overview of its usage:

```console
Usage: nat object-store s3 [OPTIONS] BUCKET_NAME COMMAND [ARGS]...

  S3 object store operations.

Options:
  --endpoint-url TEXT  S3 endpoint URL
  --access-key TEXT    S3 access key
  --secret-key TEXT    S3 secret key
  --region TEXT        S3 region
  --help               Show this message and exit.

Commands:
  delete  Delete files from an object store.
  upload  Upload a directory to an object store.
```

### Operations

#### Upload

The `nat object-store <bucket_name> upload --help` utility provides an overview of its usage:

```console
Usage: nat object-store <type> [type-options] <bucket_name> upload [OPTIONS] LOCAL_DIR

  Upload a directory to an object store.
```

#### Delete

The `nat object-store <type> <bucket_name> delete --help` utility provides an overview of its usage:

```console
Usage: nat object-store <type> [type-options] <bucket_name> delete [OPTIONS] KEYS...

  Delete files from an object store.
```
