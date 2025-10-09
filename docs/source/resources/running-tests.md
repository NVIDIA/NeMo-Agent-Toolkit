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

# Running Tests

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
- `NAT_CI_MYSQL_HOST`
- `NAT_CI_OPENSEARCH_URL`
- `NAT_CI_REDIS_HOST`
- `NAT_CI_S3_HOST`


### Start the Required Services

A Docker Compose YAML file is provided to start the required services located at `tests/test_data/docker-compose.services.yml`. The services at time of writing include Arize Phoenix, etcd, Milvus, MinIO, MySQL, OpenSearch, and Redis.

```bash
# The exact value isn't important, however it should be a secure password.
export MYSQL_ROOT_PASSWORD="$(pwgen -n 64 1)"

# Start the services in detached mode
docker compose -f tests/test_data/docker-compose.services.yml up -d
```

> [!NOTE]
> It can take some time for the services to start up. You can check the logs with:
> ```bash
> docker compose -f tests/test_data/docker-compose.services.yml logs --follow
> ```

### Run the Integration Tests
```bash
pytest --run_slow --run_integration
```

### Cleaning Up
To stop the services, run:
```bash
docker compose -f tests/test_data/docker-compose.services.yml down
```
