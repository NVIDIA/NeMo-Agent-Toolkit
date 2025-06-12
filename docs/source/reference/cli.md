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

While the AIQ toolkit library provides the capability to implement components that come together to form Agentic AI
workflow, the command line interface (CLI) provides a no code entrypoint to configure settings, access the features of
pre-built components, and mechanisms to launch workflows from configuration files. This document describes the layout
and functionality of the AIQ toolkit CLI. To begin, the command hierarchy is depicted below. Each command will be introduced
throughout the remainder of this document.

```
aiq
├── configure
│   └── channel
│       ├── add
│       ├── remove
│       └── update
├── eval
├── info
│   ├── channels
│   └── components
├── registry
│   ├── publish
│   ├── pull
│   ├── remove
│   └── search
├── run
├── serve
├── start
│   ├── console
│   ├── fastapi
│   └── mcp
├── uninstall
├── validate
└── workflow
    ├── create
    ├── reinstall
    └── delete
```

## Start

The group of `aiq start` commands provide several mechanisms to launch workflows. Each of these commands are summarized
in the following sections.

### FastAPI

The `aiq start fastapi` command will serve a FastAPI endpoint for the workflow based on the supplied configuration file
in the `--config_file` option. This command is ideal for serving a workflow as a microservice that allows client
applications to submit requests to a workflow. The `aiq serve` command is a good option when deploying this workflow into
production as the entrypoint of a containerized application. Additional options are available to serve this workflow
are made available via the `aiq start fastapi --help` utility:

```console
$ aiq start fastapi --help
Usage: aiq start fastapi [OPTIONS]

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
                                  app which run functions within the AIQ toolkit
                                  configuration. Each endpoint must have a
                                  unique path.
  --use_gunicorn BOOLEAN          Use Gunicorn to run the FastAPI app
  --runner_class TEXT             The AIQ toolkit runner class to use when launching
                                  the FastAPI app from multiple processes.
                                  Each runner is responsible for loading and
                                  running the AIQ toolkit workflow. Note: This is
                                  different from the worker class used by
                                  Gunicorn.
  --help                          Show this message and exit.
```

Once a workflow has been launched using the `aiq start fastapi` command, client applications may submit POST requests
that will run data through the hosted workflow. To access documentation on the available routes and schemas, Swagger API
documentation are made available at the <HOST>:<IP>/docs endpoint. For example, if serving locally, with
the following command:


```bash
aiq start fastapi --config_file=path/to/config --host 0.0.0.0 --port 8000
```

The Swagger API docs will be available at: [http://localhost:8000/docs](http://localhost:8000/docs)

### Console

The `aiq start console` command will run an AIQ toolkit workflow from a provided configuration file against inputs supplied
at the command line or from file using the `--inputs` and `--input_file` options, respectively. Additionally, fields in
the configuration file can be overridden by command line using the `--override` flag and dot notation to traverse to the
configuration hierarchy to the field being overridden. The run command can be useful running one off tests when
debugging a workflow. When invoking the run command, the workflow will follow the same harness as the
other workflow launch commands. This simplifies the debugging process when transitioning from development to production.

The `aiq start console` help utility provides a brief description of each option to describe is usage.

```console
$ aiq start console --help
Usage: aiq start console [OPTIONS]

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

### MCP

The `aiq start mcp` command (or simply `aiq mcp`) will start a Model Context Protocol (MCP) server that exposes workflow functions as MCP tools. This allows other applications that support the MCP protocol to use your AIQ toolkit functions directly. MCP is an open protocol developed by Anthropic that standardizes how applications provide context to LLMs. The MCP front-end is especially useful for integrating AIQ toolkit workflows with MCP-compatible clients.

The MCP front-end can be configured using the following options:

```console
$ aiq mcp --help
Usage: aiq mcp [OPTIONS]

Options:
  --config_file FILE         A JSON/YAML file that sets the parameters for the
                             workflow.  [required]
  --override <TEXT TEXT>...  Override config values using dot notation (e.g.,
                             --override llms.nim_llm.temperature 0.7)
  --name TEXT                Name of the MCP server
  --host TEXT                Host to bind the server to
  --port INTEGER             Port to bind the server to
  --debug BOOLEAN            Enable debug mode
  --log_level TEXT           Log level for the MCP server
  --tool_names TEXT          Comma-separated list of tool names to expose.
                             If not provided, all functions will be exposed.
  --help                     Show this message and exit.
```

For example, to start an MCP server with a specific workflow and expose only a particular tool:

```bash
aiq mcp --config_file examples/simple_rag/configs/milvus_rag_config.yml --tool_names mcp_retriever_tool
```

This will start an MCP server exposing the `mcp_retriever_tool` function from the workflow, which can then be accessed by any MCP-compatible client.

## Run

The `aiq run` is an alias for the `aiq start console` command and will run an AIQ toolkit workflow from a provided configuration file against inputs supplied at the
command line or from file using the `--inputs` and `--input_file` options, respectively. Additionally, fields in the
configuration file can be overridden by command line using the `--override` flag and dot notation to traverse to the
configuration hierarchy to the field being overridden. The run command can be useful running one off tests when
debugging a workflow. When invoking the run command, the workflow will follow the same harness as the
other workflow launch commands. This simplifies the debugging process when transitioning from development to production.

The `aiq run` help utility provides a brief description of each option to describe is usage.

```console
$ aiq run --help
Usage: aiq run [OPTIONS]

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
The `aiq serve` is an alias for the `aiq start fastapi` command and will serve a FastAPI endpoint for the workflow based
on the supplied configuration file in the `--config_file` option. This command is ideal for serving a workflow as a
microservice that allows client applications to submit requests to a workflow. The `aiq serve` command is a good option
when deploying this workflow into production as the entrypoint of a containerized application. Additional options are
available to serve this workflow are made available via the `aiq serve --help` utility:

```console
$ aiq serve --help
Usage: aiq serve [OPTIONS]

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
                                  app which run functions within the AIQ toolkit
                                  configuration. Each endpoint must have a
                                  unique path.
  --use_gunicorn BOOLEAN          Use Gunicorn to run the FastAPI app
  --runner_class TEXT             The AIQ toolkit runner class to use when launching
                                  the FastAPI app from multiple processes.
                                  Each runner is responsible for loading and
                                  running the AIQ toolkit workflow. Note: This is
                                  different from the worker class used by
                                  Gunicorn.
  --help                          Show this message and exit.
```

Once a workflow has been launched using the `aiq serve` command, client applications may submit POST requests that will
run data through the hosted workflow. To access documentation on the available routes and schemas, Swagger API
documentation are made available at the <HOST>:<IP>/docs endpoint. For example, if serving locally, with
the following command:

```bash
aiq serve --config_file=path/to/config --host 0.0.0.0 --port 8000
```

The Swagger API docs will be available at: [http://localhost:8000/docs](http://localhost:8000/docs)

## Evaluation
The `aiq eval` command provides access a set of evaluators designed to assessing the accuracy of AIQ toolkit workflows as
well as instrumenting their performance characteristics. Please reference
[Evaluating AIQ toolkit Workflows](../workflows/evaluate.md) for a detailed overview of the
suite of evaluation capabilities.

The `aiq eval --help` utility provides a brief overview of the command and its available options.

```console
$ aiq eval --help
Usage: aiq eval [OPTIONS] COMMAND [ARGS]...

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

## Uninstall

When a package and its corresponding components are no longer needed, they can be removed from the local environment.
This can help if certain packages are creating dependency conflicts. To remove packages from the local environment, use
the `aiq uninstall` command. This command can be used with one or more packages. The `aiq uninstall --help` utility
illustrates is usage:

```console
$ aiq uninstall --help
Usage: aiq uninstall [OPTIONS] PACKAGES COMMAND [ARGS]...

  Uninstall an AIQ toolkit plugin packages from the local environment.

Options:
  --help  Show this message and exit.
```

## Validate

Running an AIQ toolkit workflow from the CLI requires a valid workflow configuration file. Use the `aiq validate` command to
ensure a configuration files has been created with the right settings, components and parameters. It can be useful to
each components valid configuration settings using the `aiq info components` command and corresponding filters.
The `aiq validate` help utility illustrates its usage.

```console
$ aiq validate --help
Usage: aiq validate [OPTIONS]

  Validate a configuration file

Options:
  --config_file FILE  Configuration file to validate  [required]
  --help              Show this message and exit.
```

## Workflow

The extensibility of AIQ toolkit is made possible through its plugin system. To install these plugins, they must be part of
a Python package that gets installed in an environment where the AIQ toolkit library is installed. Creating boiler plate
package files (e.g. `pyproject.toml`) and component code scaffolding can be tedious. This section provides an overview
of commands that automate some of these steps.

### Create

The `aiq workflow create` command generates a valid `pyproject.toml` file with a plugin section that points to a
register.py file that has been pre-populated with AIQ toolkit programming model boiler plate code. This boiler plate code
should be further customized to implement the desired custom workflow and necessary AIQ toolkit components. The
`aiq workflow create --help` utility provides a description of its usage.

```console
$ aiq workflow create --help
Usage: aiq workflow create [OPTIONS] WORKFLOW_NAME

  Create a new AIQ toolkit workflow using templates.

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
                            components using 'aiq info component'  [default:
                            AIQ toolkit function template. Please update the
                            description.]
  --help                    Show this message and exit.
```

Also, a configuration file will be generated when you run the `aiq workflow create` command. To launch the new workflow from the CLI
(e.g. using `aiq run` or `aiq serve`), you will need a configuration file that maps to these component
configuration objects. For more information on configuration objects, refer to
[Workflow Configuration](../workflows/workflow-configuration.md).

### Reinstall

When you modify a workflow's code or update its dependencies, you need to reinstall the workflow package to ensure the changes take effect. The `aiq workflow reinstall` command rebuilds and reinstalls the workflow package with any updates. This is particularly useful after:

- Modifying the workflow's Python code
- Updating dependencies in `pyproject.toml`
- Making changes to the workflow's configuration
- Adding new tools or components

The `aiq workflow reinstall --help` utility provides a description of its usage:

```console
$ aiq workflow reinstall --help
Usage: aiq workflow reinstall [OPTIONS] WORKFLOW_NAME

  Reinstall an AIQ toolkit workflow package.

  Args:
      workflow_name (str): The name of the workflow to reinstall.

Options:
  --help  Show this message and exit.
```

For example, after updating the dependencies in your workflow's `pyproject.toml`, you would run:

```bash
aiq workflow reinstall my_workflow
```

After running the `aiq workflow reinstall` command, the following actions will happen:
1. Rebuild the workflow package
2. Uninstall the existing version
3. Install the updated version
4. Verify the installation by checking the registered components

:::{note}
If you want to completely remove a workflow instead of reinstalling it, use the `aiq workflow delete` command.
:::

### Delete

By default, unless the `--no-install` flag is set, the `aiq workflow create` command will install the generated package
into the local environment. To remove a workflow package from the local environment, use the `aiq workflow delete` command.

```console
$ aiq workflow delete --help
Usage: aiq workflow delete [OPTIONS] WORKFLOW_NAME

  Delete an AIQ toolkit workflow and uninstall its package.

  Args:     workflow_name (str): The name of the workflow to delete.

Options:
  --help  Show this message and exit.
```


## Information Commands

The `aiq info` command group provides utilities that facilitate the discovery of registered AIQ toolkit components and
retrieval of information about the locally configured AIQ toolkit environment.

### Components Information

When defining an AIQ toolkit workflow's configuration file, it can be helpful to discover the locally registered components,
possible configuration settings, and their default values. The `aiq info components` will provide this information in
tabular format with the following columns.

- `package`: The Python package containing this row's AIQ toolkit component.
- `version`: The version of the Python package containing the AIQ toolkit component.
- `component_type`: The type of AIQ toolkit component this row represents
(e.g. `front_end`, `function`, `tool_wrapper`, `llm_provider`, `llm_client`, `embedder_provider`, `embedder_client`,
`evaluator`, `memory`, `retriever_provider`, `retriever_client`, `registry_handler`, `package`).
- `component_name`: The name of the AIQ toolkit component to be specified in the `_type` field of the component's section
of the configuration file.
- `description`: A description of the component's uses, configuration parameters, and any default values. These
parameters are what will need to be specified in the configuration object.

The `aiq info components --help` utility provides an overview of usage and filter options:

```console
$ aiq info components --help
Usage: aiq info components [OPTIONS] COMMAND [ARGS]...

  List the locally registered AIQ toolkit components.

Options:
  -t, --types [front_end|function|tool_wrapper|llm_provider|llm_client|embedder_provider|embedder_client|evaluator|memory|retriever_provider|retriever_client|registry_handler|logging|tracing|package|undefined]
                                  Filter the search by AIQ toolkit component type.
  -o, --output_path TEXT          Path to save search results.
  -q, --query TEXT                The query string.  [default: ""]
  -n, --num_results INTEGER       Number of results to return.  [default: -1]
  -f, --fields [all|package|version|component_name|description|developer_notes]
                                  Fields used when applying query.
  --help                          Show this message and exit.
```

### Channels Information

The `aiq info channels` command provides a list of each configured remote registry channel and their corresponding
configuration settings. This command provides the `-t, --type` option to filter the remote registry channels by type.
By default, this command will return an empty list. The `aiq registry` command group will not be functional without
first configuring registry channels with the `aiq configure channel add` command. Successful channel configurations
will be returned when invoking the `aiq info channels` command.

The `aiq info channels --help` provides an overview of its usage:

```console
$ aiq info channels --help
Usage: aiq info channels [OPTIONS] COMMAND [ARGS]...

  List the configured remote registry channels.

Options:
  -t, --type TEXT  Filter the results by channel type.
  --help           Show this message and exit.
```

## Configuration Commands

An AIQ toolkit developer may want to configure persistent settings for their development environment. These settings would
be configured once to setup their development environment so they can focus on software development from that point
forward. This section discusses the various configuration settings available for AIQ toolkit developers.

### Remote Registry Configuration

One of the core value propositions of the AIQ toolkit library is the redistribution of components with other developers.
Being able to package and distribute packages such that other developers can leverage them is critical to accelerating
developer velocity. Similarly, being able to discover and install components built by others will improve the
current developer's velocity. To facilitate this process, AIQ toolkit implements a remote registry `channel` concept that
allows AIQ toolkit developers to subscribe to registries that store published AIQ toolkit packages, each container containing
usable components. A `channel` is analogous to a Conda channel for Anaconda users or a PyPI registry for pip users.


#### Adding a Remote Registry Channel
Currently, there are two channel types that facilitate remote discovery and reuse:
 - `rest` – provides a contract driven interface to a registry service behind a REST endpoint
 - `pypi` – a simple interface to publish packages to a private PyPI registry.

Invoking the `aiq info components` command provides a description of the available channel settings.

Here we provide a example that configures a remote rest channel. To use this channel, there must exists a remote
registry that adheres to the contracts defined in the rest handler in AIQ toolkit.

```console
$ aiq configure channel add rest
Channel Name: my_rest_channel  # A user defined locally unique name used to reference this configured channel
Endpoint: http://my_rest_channel_url.com  # The endpoint to the remote rest registry service
Token: my_rest_token  # The authentication token to interact with this rest registry service
Publish Route: publish  # The route to use when publishing AIQ toolkit packages
Pull Route: pull  # The route to use when downloading AIQ toolkit packages
Search Route: search  # The route use when searching for relevant AIQ toolkit packages
Remove Route: remove  # The route to use when removing a published package from a remote rest registy
```

Here we provide a example that configures a remote `pypi` channel. This assumes there exists a private PyPI registry.

```console
$ aiq configure channel add pypi
Channel Name: my_pypi_channel  # A user defined locally unique name used to reference this configured channel
Endpoint: http://my_pypi_channel_url.com  # The endpoint to the private pypi registry service
Token: my_pypi_token  # The authentication token to interact with this pypi registry service
Publish Route:  # The route to use when publishing AIQ toolkit packages, setting an empty value here
Pull Route: # The route to use when downloading AIQ toolkit packages, setting an empty value here
Search Route: simple  # The route use when searching for relevant AIQ toolkit packages
```

#### Updating a Remote Registry Channel Configuration

At some point, a developer might need to update a remote registry channel's configuration settings. In this case,
using the `aiq configure channel update` command will select a remote registry channel by its locally unique name and allow
the developer to override the configuration settings.

A usage example is provided below:

```console
$ aiq configure channel update my_rest_channel
Endpoint: http://my_updated_rest_channel_url.com  # The overridden endpoint to the remote rest registry service
Token: my_rest_token
Publish Route: publish
Pull Route: pull
Search Route: search
Remove Route: remove
```

#### Removing a Remote Registry Channel

A developer may need to remove a locally configured remote registry channel. In this case, the `aiq registry remove`
command can be used. The channel will be removed based on the name supplied with the command.

An example of using this command is provided below:

```bash
aiq configure channel remove my_rest_channel
```

Note, once a channel is removed, it will no longer be able to support `aiq registry publish`, `aiq registry search`,
`aiq registry pull`, or `aiq registry remove` commands until reconfigured.

## Remote Registry Interactions

AIQ toolkit is designed to be a community oriented library. This means that developer productivity is maximized when others
distribute AIQ toolkit plugin packages that will benefit others. This section will introduce the mechanisms the AIQ toolkit CLI
exposes to facilitate publishing, discovering, downloading, and removing AIQ toolkit packages from a configured remote
registry. Here we define a remote registry as a centralized location that stores plugin wheel packages and AIQ toolkit
specific metadata to that describes its usage details. Before these commands can be used, a remote registry must be
available and a developer must have configured the corresponding channel using the `aiq configure channel add` command.
Refer to [Adding a Remote Registry Channel](#adding-a-remote-registry-channel) for more details on adding a remote registry channels.

The `aiq registry` help command will provide the available commands in this group.

```console
$ aiq registry --help
Usage: aiq registry [OPTIONS] COMMAND [ARGS]...

  Utility to configure AIQ toolkit remote registry channels.

Options:
  --help  Show this message and exit.

Commands:
  publish  Publish local AIQ toolkit artifacts to a remote registry from package...
  pull     Pull AIQ toolkit artifacts from a remote registry by package name.
  remove   Remove AIQ toolkit artifact from a remote registry by name and version.
  search   Search for AIQ toolkit artifacts from remote registry.
```

#### Publishing AIQ Toolkit Components

AIQ toolkit developers may want to distribute their components with the broader ecosystem. The AIQ toolkit publish CLI utility
provides a mechanism to publish an AIQ toolkit plugin package to a remote registry channel so that other developers can
benefit from it's implemented components. Invoking the `aiq registry publish` command will build a package wheel, gather
all component metadata, and transmit to the specified remote registry by channel name. Note, a package must be first
installed locally so the discovery hooks can pull in necessary AIQ toolkit component metadata.

The `aiq registry publish --help` utility provides an overview of its usage:

```console
$ aiq registry publish --help
Usage: aiq registry publish [OPTIONS] PACKAGE_ROOT COMMAND [ARGS]...

  Publish local AIQ toolkit artifacts to a remote registry from package
  repository.

Options:
  --config_file FILE  A YAML file to override configured channel settings.
  -c, --channel TEXT  The remote registry channel to use when publishing the
                      AIQ toolkit artifact.  [required]
  --help              Show this message and exit.

```

#### Discovering AIQ Toolkit Components

When developing and deploying AIQ toolkit workflows, it is most efficient to leverage pre-built components. When using
pre-built components will, only configuration settings are required to integration with the rest of a workflow. These
pre-built exist in the core library, as well as, within other AIQ toolkit plugin packages. Remote registry channels are the
formal mechanism to publish reusable components to the community. The `aiq registry search` command allows developers
to search relevant pre-built components that might benefit their application. The search command is usually followed up
by an `aiq registry pull` command, once a useful package has been identified.

The `aiq registry search --help` utility provides an overview of its usage:

```console
$ aiq registry search --help
Usage: aiq registry search [OPTIONS] COMMAND [ARGS]...

  Search for AIQ toolkit artifacts from remote registry.

Options:
  --config_file FILE              A JSON/YAML file that sets the parameters
                                  for the workflow.
  -c, --channel TEXT              The remote registry channel to use when
                                  pulling the AIQ toolkit artifact.  [required]
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

#### Pulling in AIQ Toolkit Components
Once a useful AIQ toolkit component has been discovered using the `aiq registry search` command, the containing package can be
pulled in and installed from a configured remote registry, so that it can be used withing the local AIQ toolkit environment.
Once installed, all components in the package can be referenced by name in an AIQ toolkit workflow YAML configuration file.
In many cases, components can be stitched together in YAML without having to write much integration code.

The `aiq registry pull --help` command provides an overview of its usage:

```console
$ aiq registry pull --help
Usage: aiq registry pull [OPTIONS] PACKAGES COMMAND [ARGS]...

  Pull AIQ toolkit artifacts from a remote registry by package name.

Options:
  --config_file FILE  A YAML file to override the channel settings.
  -c, --channel TEXT  The remote registry channel to use when pulling the
                      AIQ toolkit artifact.  [required]
  --help              Show this message and exit.
```

Note, the supplied package takes the following format: `package_name==version`, where the package version is optional.


#### Removing AIQ Toolkit Components
In rare cases, it might make sense to remove a package from a remote registry over a configured remote registry channel.
This the `aiq registry remove` command provides support for this feature, assuming the remote registry provides and
allows this interaction.

The `aiq registry remove --help` utility provides an overview of its usage.

```console
$ aiq registry remove --help
Usage: aiq registry remove [OPTIONS] PACKAGES COMMAND [ARGS]...

  Remove AIQ toolkit artifact from a remote registry by name and version.

Options:
  --config_file FILE  A YAML file to override the channel settings.
  -c, --channel TEXT  The remote registry channel that will remove the AIQ toolkit
                      artifact.  [required]
  --help              Show this message and exit.
```
