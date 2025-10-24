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

# Running NVIDIA NeMo Agent Toolkit Tests

NeMo Agent toolkit uses [pytest](https://docs.pytest.org/en/stable) for running tests. To run the basic set of tests, from the root of the repository, run:
```bash
pytest
```

## Optional pytest Flags
NeMo Agent toolkit adds the following optional pytest flags to control which tests are run:

| Flag | Description |
|------|-------------|
| `--run_slow` | Run tests marked as slow, these tests typically take longer than 30 seconds to run. |
| `--run_integration` | Run tests marked as integration, these tests typically require external services, and may require an API key. |
| `--fail_missing` | Typically tests which require a service to be running or a specific API key will be skipped if the service isn't available or the API key is not set. When the `--fail_missing` flag is set, these tests will be marked as failed instead of skipped, this is useful when debugging a specific test. |


## Running Integration Tests

Running the integration tests requires several services to be running, and several API keys to be set. However by default the integration tests are skipped if the required services or API keys are not available.

### Set the API keys:
```bash
export AWS_ACCESS_KEY_ID=<KEY>
export AWS_SECRET_ACCESS_KEY=<KEY>
export AZURE_OPENAI_API_KEY=<KEY>
export MEM0_API_KEY=<KEY>
export NVIDIA_API_KEY=<KEY>
export OPENAI_API_KEY=<KEY>
export SERP_API_KEY=<KEY>  # https://serpapi.com
export SERPERDEV_API_KEY=<KEY>  # https://serper.dev
export TAVILY_API_KEY=<KEY>
```

### Optional variables
```bash
export AZURE_OPENAI_DEPLOYMENT="<custom model>"
export AZURE_OPENAI_ENDPOINT="<your-custom-endpoint>"
```

### Other Variables
- `NAT_CI_ETCD_HOST`
- `NAT_CI_ETCD_PORT`
- `NAT_CI_MILVUS_HOST`
- `NAT_CI_MILVUS_PORT`
- `NAT_CI_MINIO_HOST`
- `NAT_CI_MYSQL_HOST`
- `NAT_CI_OPENSEARCH_URL`
- `NAT_CI_PHOENIX_URL`
- `NAT_CI_REDIS_HOST`


### Start the Required Services

A Docker Compose YAML file is provided to start the required services located at `tests/test_data/docker-compose.services.yml`. The services at time of writing include Arize Phoenix, etcd, Milvus, MinIO, MySQL, OpenSearch, and Redis.

```bash
# Create temporary passwords for the services
function mk_pw() {
  pwgen -n 64 1
}

export CLICKHOUSE_PASSWORD="$(mk_pw)"
export LANGFUSE_NEXTAUTH_SECRET="$(mk_pw)"
export LANGFUSE_PUBLIC_KEY="lf_pk_$(mk_pw)"
export LANGFUSE_SALT="$(mk_pw)"
export LANGFUSE_SECRET_KEY="lf_sk_$(mk_pw)"
export LANGFUSE_USER_PW="$(mk_pw)"
export MYSQL_ROOT_PASSWORD="$(mk_pw)"
export POSTGRES_PASSWORD="$(mk_pw)"

# Start the services in detached mode
docker compose -f tests/test_data/docker-compose.services.yml up -d
```

:::{note}
It can take some time for the services to start up. You can check the logs with:
```bash
docker compose -f tests/test_data/docker-compose.services.yml logs --follow
```
:::

### Run the Integration Tests
```bash
pytest --run_slow --run_integration
```

### Cleaning Up
To stop the services, run:
```bash
docker compose -f tests/test_data/docker-compose.services.yml down
```

## Writing Integration Tests
Many of the example workflows cannot be fully tested with unit tests alone, as they typically require an actual LLM service and potentially other services to be running.

### Typical Example of an Integration Test

`examples/frameworks/multi_frameworks/tests/test_multi_frameworks_workflow.py`:
```python
import pytest


@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.usefixtures("nvidia_api_key")
async def test_full_workflow():
    from nat.test.utils import locate_example_config
    from nat.test.utils import run_workflow
    from nat_multi_frameworks.register import MultiFrameworksWorkflowConfig

    config_file = locate_example_config(MultiFrameworksWorkflowConfig)

    await run_workflow(config_file=config_file, question="tell me about this workflow", expected_answer="workflow")
```

In the above example, the `@pytest.mark.integration` decorator marks the test as an integration test, this will cause the test to be skipped unless the `--run_integration` flag is provided when running pytest. Similarly the `@pytest.mark.slow` decorator marks the test as a slow test, which will be skipped unless the `--run_slow` flag is provided.

The workflow being run requires a valid NVIDIA API key to be set in the `NVIDIA_API_KEY` environment variable, the `@pytest.mark.usefixtures("nvidia_api_key")` decorator ensures that the test is skipped if the API key is not set. This fixture along with many others are defined in `packages/nvidia_nat_test/src/nat/test/plugin.py`, and are available for use in tests if the `nvidia-nat-test` package is installed. Most of the API keys used in NeMo Agent toolkit workflows have corresponding fixtures defined there (for example: `openai_api_key`, `tavily_api_key`, `mem0_api_key`, and others).

The `locate_example_config` utility function is used to locate the configuration file relative to the configuration class. By default this function searches for a file named `config.yml`, alternately the `config_file` argument can be specified (ex: `locate_example_config(RetryReactAgentConfig, "config-hitl.yml")`). This function will work with any workflow that has the same layout structure as a workflow created using the `nat workflow create` command. This function works for both example workflows in the NeMo Agent toolkit repository itself, and workflows in another repository that has the `nvidia-nat-test` installed.

The `run_workflow` utility function is used to run the workflow with the specified configuration file, question, and expected answer. Since the results of LLM calls can vary it is best to use simple questions and expected answers that are likely to be returned consistently. By default a case-insensitive match is used for the `expected_answer`. Alternately the `assert_expected_answer` parameter can be set to `False` allowing the test to perform custom validation of the result returned by the workflow:
```python
    result = await run_workflow(config_file=config_file,
                                question="What are LLMs?",
                                expected_answer="",
                                assert_expected_answer=False)
    assert re.match(r".*large language model.*", result, re.IGNORECASE) is not None
```

#### Workflows Without Configuration Classes
If the workflow being tested contains only a YAML, the configuration file can be located relative to the root of the repository using the  `root_repo_dir` fixture or relative to the `examples/` directory using the `examples_dir` fixture.

`examples/agents/tests/test_agents.py`:
<!-- path-check-skip-begin -->
```python
@pytest.mark.integration
@pytest.mark.usefixtures("nvidia_api_key")
async def test_react_agent_full_workflow(examples_dir: Path):
    config_file = examples_dir / "agents/react/configs/config.yml"
    await run_workflow(config_file=config_file, question="What are LLMs?", expected_answer="Large Language Model")
```

:::{note}
While most of the fixtures defined in the `nvidia-nat-test` package are available for use in tests in third-party packages, a few such as `root_repo_dir` and `examples_dir` only function correctly when used within the NeMo Agent toolkit repository itself. As an alternative, a configuration file can be located relative to the test file using: `config_file = Path(__file__).parent / "configs/config.yml"`.
:::
<!-- path-check-skip-end -->

#### Workflows Requiring a Service
Many of the existing services that NeMo Agent toolkit workflows can interact with have corresponding fixtures defined in the `nvidia-nat-test` package to ensure that the service is running before the test is run, these are defined in `packages/nvidia_nat_test/src/nat/test/plugin.py`.

A typical example of such a fixture is the `milvus_uri` fixture, which ensures that the Milvus service is running and provides the URL to connect to it:
```python
@pytest.fixture(name="milvus_uri", scope="session")
def milvus_uri_fixture(etcd_url: str, fail_missing: bool = False) -> str:
    """
    To run these tests, a Milvus server must be running
    """
    host = os.getenv("NAT_CI_MILVUS_HOST", "localhost")
    port = os.getenv("NAT_CI_MILVUS_PORT", "19530")
    uri = f"http://{host}:{port}"
    try:
        from pymilvus import MilvusClient
        MilvusClient(uri=uri)

        return uri
    except:  # noqa: E722
        reason = f"Unable to connect to Milvus server at {uri}"
        if fail_missing:
            raise RuntimeError(reason)
        pytest.skip(reason=reason)
```

The above fixture is scoped to the session, ensuring it will only be run once per test session. The `pymilvus` library is imported lazily within the body of the fixture, this avoids unnecessary imports to be performed during test collection. This is especially important in this case as the `pymilvus` library is an optional dependency of NeMo Agent toolkit, and may not be installed in all environments. Since the import is performed within a try/except block, if the library is not installed the test will be skipped (unless the user also ran pytest with the `--fail_missing` flag).

Of note is that the host and port of the service can be configured via environment variables, this allows the test to connect to services running in different environments.

The Milvus service requires an instance of the etcd service to be running, so the `etcd_url` fixture is included as a dependency, ensuring that the etcd service is running before attempting to connect to Milvus.

An example of a test using the `milvus_uri` fixture is shown below:
`examples/custom_functions/automated_description_generation/tests/test_auto_desc_generation.py`
```python
@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.usefixtures("nvidia_api_key", "populate_milvus")
async def test_full_workflow(milvus_uri: str) -> None:
    from pydantic import HttpUrl

    from nat.runtime.loader import load_config
    from nat.test.utils import locate_example_config
    from nat.test.utils import run_workflow
    from nat_automated_description_generation.register import AutomatedDescriptionMilvusWorkflow

    config_file = locate_example_config(AutomatedDescriptionMilvusWorkflow)
    config = load_config(config_file)
    config.retrievers['retriever'].uri = HttpUrl(url=milvus_uri)

    # Unfortunately the workflow itself returns inconsistent results
    await run_workflow(config=config, question="List 5 subspecies of Aardvark?", expected_answer="Aardvark")
```

Of note here is that an additional fixture `populate_milvus` is used to ensure that the Milvus instance is populated with test data before this test is run. Additionally the `examples/custom_functions/automated_description_generation/configs/config.yml` configuration file specifies a Milvus URL of `http://localhost:19530`, which is replaced at runtime with the actual URL provided by the `milvus_uri` fixture. This allows the test to run against a Milvus instance running in a different environment if needed.

Finally the new service should be added to the Docker Compose YAML file located at `tests/test_data/docker-compose.services.yml` to allow easy startup of the service when running integration tests locally.
