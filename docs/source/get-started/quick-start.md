# Quick Start with NVIDIA NeMo Agent Toolkit

This guide will walk you through running and evaluating existing workflows.

## Obtaining API Keys
Depending on which workflows you are running, you may need to obtain API keys from the respective services. Most NeMo Agent toolkit workflows require an NVIDIA API key defined with the `NVIDIA_API_KEY` environment variable. An API key can be obtained by visiting [`build.nvidia.com`](https://build.nvidia.com/) and creating an account.

### Optional OpenAI API Key
Some workflows may also require an OpenAI API key. Visit [OpenAI](https://openai.com/) and create an account. Navigate to your account settings to obtain your OpenAI API key. Copy the key and set it as an environment variable using the following command:

```bash
export OPENAI_API_KEY="<YOUR_OPENAI_API_KEY>"
```

## Running Example Workflows

Before running any of the NeMo Agent toolkit examples, set your NVIDIA API key as an
environment variable to access NVIDIA AI services.

```bash
export NVIDIA_API_KEY=<YOUR_API_KEY>
```

:::{note}
Replace `<YOUR_API_KEY>` with your actual NVIDIA API key.
:::

### Running the Simple Workflow

1. Install the `nat_simple_web_query` Workflow

    ```bash
    uv pip install -e examples/getting_started/simple_web_query
    ```

2. Run the `nat_simple_web_query` Workflow

    ```bash
    nat run --config_file=examples/getting_started/simple_web_query/configs/config.yml --input "What is LangSmith"
    ```

3. **Run and evaluate the `nat_simple_web_query` Workflow**

    The `eval_config.yml` YAML is a super-set of the `config.yml` containing additional fields for evaluation. To evaluate the `nat_simple_web_query` workflow, run the following command:
    ```bash
    nat eval --config_file=examples/evaluation_and_profiling/simple_web_query_eval/configs/eval_config.yml
    ```


## NeMo Agent Toolkit Packages
Once a NeMo Agent toolkit workflow is ready for deployment to production, the deployed workflow will need to declare a dependency on the `nvidia-nat` package, along with the needed plugins. When declaring a dependency on NeMo Agent toolkit it is recommended to use the first two digits of the version number. For example if the version is `1.0.0` then the dependency would be `1.0`.

For more information on the available plugins, refer to [Framework Integrations](./installing.md#framework-integrations).

Example dependency for NeMo Agent toolkit using the LangChain/LangGraph plugin for projects using a `pyproject.toml` file:
```toml
dependencies = [
"nvidia-nat[langchain]~=1.0",
# Add any additional dependencies your workflow needs
]
```

Alternately for projects using a `requirements.txt` file:
```
nvidia-nat[langchain]==1.0.*
```

## Next Steps

* Review the NeMo Agent toolkit [tutorials](../tutorials/index.md) for detailed guidance on using the toolkit.
* Explore the examples in the `examples` directory to learn how to build custom workflows and tools with NeMo Agent toolkit.
