# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import random
import subprocess
import time
import types
import typing
from collections.abc import AsyncGenerator
from collections.abc import Generator
from pathlib import Path

import pytest
import pytest_asyncio

if typing.TYPE_CHECKING:
    import galileo.log_streams
    import galileo.projects
    import langsmith.client

    from docker.client import DockerClient


def pytest_addoption(parser: pytest.Parser):
    """
    Adds command line options for running specfic tests that are disabled by default
    """
    parser.addoption(
        "--run_integration",
        action="store_true",
        dest="run_integration",
        help=("Run integrations tests that would otherwise be skipped. "
              "This will call out to external services instead of using mocks"),
    )

    parser.addoption(
        "--run_slow",
        action="store_true",
        dest="run_slow",
        help="Run end to end tests that would otherwise be skipped",
    )

    parser.addoption(
        "--fail_missing",
        action="store_true",
        dest="fail_missing",
        help=("Tests requiring unmet dependencies are normally skipped. "
              "Setting this flag will instead cause them to be reported as a failure"),
    )


def pytest_runtest_setup(item):
    if (not item.config.getoption("--run_integration")):
        if (item.get_closest_marker("integration") is not None):
            pytest.skip("Skipping integration tests by default. Use --run_integration to enable")

    if (not item.config.getoption("--run_slow")):
        if (item.get_closest_marker("slow") is not None):
            pytest.skip("Skipping slow tests by default. Use --run_slow to enable")


@pytest.fixture(name="register_components", scope="session", autouse=True)
def register_components_fixture():
    from nat.runtime.loader import PluginTypes
    from nat.runtime.loader import discover_and_register_plugins

    # Ensure that all components which need to be registered as part of an import are done so. This is necessary
    # because imports will not be reloaded between tests, so we need to ensure that all components are registered
    # before any tests are run.
    discover_and_register_plugins(PluginTypes.ALL)

    # Also import the nat.test.register module to register test-only components


@pytest.fixture(name="module_registry", scope="module", autouse=True)
def module_registry_fixture():
    """
    Resets and returns the global type registry for testing

    This gets automatically used at the module level to ensure no state is leaked between modules
    """
    from nat.cli.type_registry import GlobalTypeRegistry

    with GlobalTypeRegistry.push() as registry:
        yield registry


@pytest.fixture(name="registry", scope="function", autouse=True)
def function_registry_fixture():
    """
    Resets and returns the global type registry for testing

    This gets automatically used at the function level to ensure no state is leaked between functions
    """
    from nat.cli.type_registry import GlobalTypeRegistry

    with GlobalTypeRegistry.push() as registry:
        yield registry


@pytest.fixture(scope="session", name="fail_missing")
def fail_missing_fixture(pytestconfig: pytest.Config) -> bool:
    """
    Returns the value of the `fail_missing` flag, when false tests requiring unmet dependencies will be skipped, when
    True they will fail.
    """
    yield pytestconfig.getoption("fail_missing")


def require_env_variables(varnames: list[str], reason: str, fail_missing: bool = False) -> dict[str, str]:
    """
    Checks if the given environment variable is set, and returns its value if it is. If the variable is not set, and
    `fail_missing` is False the test will ve skipped, otherwise a `RuntimeError` will be raised.
    """
    env_variables = {}
    try:
        for varname in varnames:
            env_variables[varname] = os.environ[varname]
    except KeyError as e:
        if fail_missing:
            raise RuntimeError(reason) from e

        pytest.skip(reason=reason)

    return env_variables


@pytest.fixture(name="openai_api_key", scope='session')
def openai_api_key_fixture(fail_missing: bool):
    """
    Use for integration tests that require an Openai API key.
    """
    yield require_env_variables(
        varnames=["OPENAI_API_KEY"],
        reason="openai integration tests require the `OPENAI_API_KEY` environment variable to be defined.",
        fail_missing=fail_missing)


@pytest.fixture(name="nvidia_api_key", scope='session')
def nvidia_api_key_fixture(fail_missing: bool):
    """
    Use for integration tests that require an Nvidia API key.
    """
    yield require_env_variables(
        varnames=["NVIDIA_API_KEY"],
        reason="Nvidia integration tests require the `NVIDIA_API_KEY` environment variable to be defined.",
        fail_missing=fail_missing)


@pytest.fixture(name="serp_api_key", scope='session')
def serp_api_key_fixture(fail_missing: bool):
    """
    Use for integration tests that require a SERP API (serpapi.com) key.
    """
    yield require_env_variables(
        varnames=["SERP_API_KEY"],
        reason="SERP integration tests require the `SERP_API_KEY` environment variable to be defined.",
        fail_missing=fail_missing)


@pytest.fixture(name="serperdev", scope='session')
def serperdev_api_key_fixture(fail_missing: bool):
    """
    Use for integration tests that require a Serper Dev API (https://serper.dev) key.
    """
    yield require_env_variables(
        varnames=["SERPERDEV_API_KEY"],
        reason="SERPERDEV integration tests require the `SERPERDEV_API_KEY` environment variable to be defined.",
        fail_missing=fail_missing)


@pytest.fixture(name="tavily_api_key", scope='session')
def tavily_api_key_fixture(fail_missing: bool):
    """
    Use for integration tests that require a Tavily API key.
    """
    yield require_env_variables(
        varnames=["TAVILY_API_KEY"],
        reason="Tavily integration tests require the `TAVILY_API_KEY` environment variable to be defined.",
        fail_missing=fail_missing)


@pytest.fixture(name="mem0_api_key", scope='session')
def mem0_api_key_fixture(fail_missing: bool):
    """
    Use for integration tests that require a Mem0 API key.
    """
    yield require_env_variables(
        varnames=["MEM0_API_KEY"],
        reason="Mem0 integration tests require the `MEM0_API_KEY` environment variable to be defined.",
        fail_missing=fail_missing)


@pytest.fixture(name="aws_keys", scope='session')
def aws_keys_fixture(fail_missing: bool):
    """
    Use for integration tests that require AWS credentials.
    """

    yield require_env_variables(
        varnames=["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"],
        reason=
        "AWS integration tests require the `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY` environment variables to be "
        "defined.",
        fail_missing=fail_missing)


@pytest.fixture(name="azure_openai_keys", scope='session')
def azure_openai_keys_fixture(fail_missing: bool):
    """
    Use for integration tests that require Azure OpenAI credentials.
    """
    yield require_env_variables(
        varnames=["AZURE_OPENAI_API_KEY", "AZURE_OPENAI_ENDPOINT"],
        reason="Azure integration tests require the `AZURE_OPENAI_API_KEY` and `AZURE_OPENAI_ENDPOINT` environment "
        "variables to be defined.",
        fail_missing=fail_missing)


@pytest.fixture(name="langfuse_keys", scope='session')
def langfuse_keys_fixture(fail_missing: bool):
    """
    Use for integration tests that require Langfuse credentials.
    """
    yield require_env_variables(
        varnames=["LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY"],
        reason="Langfuse integration tests require the `LANGFUSE_PUBLIC_KEY` and `LANGFUSE_SECRET_KEY` environment "
        "variables to be defined.",
        fail_missing=fail_missing)


@pytest.fixture(name="wandb_api_key", scope='session')
def wandb_api_key_fixture(fail_missing: bool):
    """
    Use for integration tests that require a Weights & Biases API key.
    """
    yield require_env_variables(
        varnames=["WANDB_API_KEY"],
        reason="Weights & Biases integration tests require the `WANDB_API_KEY` environment variable to be defined.",
        fail_missing=fail_missing)


@pytest.fixture(name="weave", scope='session')
def require_weave_fixture(fail_missing: bool) -> types.ModuleType:
    """
    Use for integration tests that require Weave to be running.
    """
    try:
        import weave
        return weave
    except Exception as e:
        reason = "Weave must be installed to run weave based tests"
        if fail_missing:
            raise RuntimeError(reason) from e
        pytest.skip(reason=reason)


@pytest.fixture(name="langsmith_api_key", scope='session')
def langsmith_api_key_fixture(fail_missing: bool):
    """
    Use for integration tests that require a LangSmith API key.
    """
    yield require_env_variables(
        varnames=["LANGSMITH_API_KEY"],
        reason="LangSmith integration tests require the `LANGSMITH_API_KEY` environment variable to be defined.",
        fail_missing=fail_missing)


@pytest.fixture(name="langsmith_client")
def langsmith_client_fixture(langsmith_api_key: str, fail_missing: bool) -> "langsmith.client.Client":
    try:
        import langsmith.client
        client = langsmith.client.Client()
        return client
    except ImportError:
        reason = "LangSmith integration tests require the `langsmith` package to be installed."
        if fail_missing:
            raise RuntimeError(reason)
        pytest.skip(reason=reason)


@pytest.fixture(name="project_name")
def project_name_fixture() -> str:
    # Create a unique project name for each test run
    return f"nat-e2e-test-{time.time()}-{random.random()}"


@pytest.fixture(name="langsmith_project_name")
def langsmith_project_name_fixture(langsmith_client: "langsmith.client.Client", project_name: str) -> Generator[str]:
    langsmith_client.create_project(project_name)
    yield project_name

    langsmith_client.delete_project(project_name=project_name)


@pytest.fixture(name="galileo_api_key", scope='session')
def galileo_api_key_fixture(fail_missing: bool):
    """
    Use for integration tests that require a Galileo API key.
    """
    yield require_env_variables(
        varnames=["GALILEO_API_KEY"],
        reason="Galileo integration tests require the `GALILEO_API_KEY` environment variable to be defined.",
        fail_missing=fail_missing)


@pytest.fixture(name="galileo_project")
def galileo_project_fixture(galileo_api_key: str, fail_missing: bool,
                            project_name: str) -> Generator["galileo.projects.Project"]:
    """
    Creates a unique Galileo project and deletes it after the test run.
    """
    try:
        import galileo.projects
        project = galileo.projects.create_project(name=project_name)
        yield project

        galileo.projects.delete_project(id=project.id)
    except ImportError as e:
        reason = "Galileo integration tests require the `galileo` package to be installed."
        if fail_missing:
            raise RuntimeError(reason) from e
        pytest.skip(reason=reason)


@pytest.fixture(name="galileo_log_stream")
def galileo_log_stream_fixture(galileo_project: "galileo.projects.Project") -> "galileo.log_streams.LogStream":
    """
    Creates a Galileo log stream for integration tests.

    The log stream is automatically deleted when the associated project is deleted.
    """
    import galileo.log_streams
    return galileo.log_streams.create_log_stream(project_id=galileo_project.id, name="test")


@pytest.fixture(name="require_docker", scope='session')
def require_docker_fixture(fail_missing: bool) -> "DockerClient":
    """
    Use for integration tests that require Docker to be running.
    """
    try:
        from docker.client import DockerClient
        yield DockerClient()
    except Exception as e:
        reason = f"Unable to connect to Docker daemon: {e}"
        if fail_missing:
            raise RuntimeError(reason) from e
        pytest.skip(reason=reason)


@pytest.fixture(name="restore_environ")
def restore_environ_fixture():
    orig_vars = os.environ.copy()
    yield os.environ

    for key, value in orig_vars.items():
        os.environ[key] = value

    # Delete any new environment variables
    # Iterating over a copy of the keys as we will potentially be deleting keys in the loop
    for key in list(os.environ.keys()):
        if key not in orig_vars:
            del (os.environ[key])


@pytest.fixture(name="root_repo_dir", scope='session')
def root_repo_dir_fixture() -> Path:
    from nat.test.utils import locate_repo_root
    return locate_repo_root()


@pytest.fixture(name="examples_dir", scope='session')
def examples_dir_fixture(root_repo_dir: Path) -> Path:
    return root_repo_dir / "examples"


@pytest.fixture(name="env_without_nat_log_level", scope='function')
def env_without_nat_log_level_fixture() -> dict[str, str]:
    env = os.environ.copy()
    env.pop("NAT_LOG_LEVEL", None)
    return env


@pytest.fixture(name="etcd_url", scope="session")
def etcd_url_fixture(fail_missing: bool = False) -> str:
    """
    To run these tests, an etcd server must be running
    """
    import requests

    host = os.getenv("NAT_CI_ETCD_HOST", "localhost")
    port = os.getenv("NAT_CI_ETCD_PORT", "2379")
    url = f"http://{host}:{port}"
    health_url = f"{url}/health"

    try:
        response = requests.get(health_url, timeout=5)
        response.raise_for_status()
        return url
    except:  # noqa: E722
        failure_reason = f"Unable to connect to etcd server at {url}"
        if fail_missing:
            raise RuntimeError(failure_reason)
        pytest.skip(reason=failure_reason)


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


@pytest.fixture(name="populate_milvus", scope="session")
def populate_milvus_fixture(milvus_uri: str, root_repo_dir: Path):
    """
    Populate Milvus with some test data.
    """
    populate_script = root_repo_dir / "scripts/langchain_web_ingest.py"

    # Ingest default cuda docs
    subprocess.run(["python", str(populate_script), "--milvus_uri", milvus_uri], check=True)

    # Ingest MCP docs
    subprocess.run([
        "python",
        str(populate_script),
        "--milvus_uri",
        milvus_uri,
        "--urls",
        "https://github.com/modelcontextprotocol/python-sdk",
        "--urls",
        "https://modelcontextprotocol.io/introduction",
        "--urls",
        "https://modelcontextprotocol.io/quickstart/server",
        "--urls",
        "https://modelcontextprotocol.io/quickstart/client",
        "--urls",
        "https://modelcontextprotocol.io/examples",
        "--urls",
        "https://modelcontextprotocol.io/docs/concepts/architecture",
        "--collection_name",
        "mcp_docs"
    ],
                   check=True)

    # Ingest some wikipedia docs
    subprocess.run([
        "python",
        str(populate_script),
        "--milvus_uri",
        milvus_uri,
        "--urls",
        "https://en.wikipedia.org/wiki/Aardvark",
        "--collection_name",
        "wikipedia_docs"
    ],
                   check=True)


@pytest.fixture(name="require_nest_asyncio", scope="session")
def require_nest_asyncio_fixture():
    """
    Some tests require nest_asyncio to be installed to allow nested event loops, calling nest_asyncio.apply() more than
    once is a no-op so it's safe to call this fixture even if one of our dependencies already called it.
    """
    import nest_asyncio
    nest_asyncio.apply()


@pytest.fixture(name="phoenix_url", scope="session")
def phoenix_url_fixture(fail_missing: bool) -> str:
    """
    To run these tests, a phoenix server must be running.
    The phoenix server can be started by running the following command:
    docker run -p 6006:6006 -p 4317:4317  arizephoenix/phoenix:latest
    """
    import requests

    url = os.getenv("NAT_CI_PHOENIX_URL", "http://localhost:6006")
    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status()

        return url
    except Exception as e:
        reason = f"Unable to connect to Phoenix server at {url}: {e}"
        if fail_missing:
            raise RuntimeError(reason)
        pytest.skip(reason=reason)


@pytest.fixture(name="phoenix_trace_url", scope="session")
def phoenix_trace_url_fixture(phoenix_url: str) -> str:
    """
    Some of our tools expect the base url provided by the phoenix_url fixture, however the
    general.telemetry.tracing["phoenix"].endpoint expects the trace url which is what this fixture provides.
    """
    return f"{phoenix_url}/v1/traces"


@pytest.fixture(name="redis_server", scope="session")
def fixture_redis_server(fail_missing: bool) -> Generator[dict[str, str | int]]:
    """Fixture to safely skip redis based tests if redis is not running"""
    host = os.environ.get("NAT_CI_REDIS_HOST", "localhost")
    port = int(os.environ.get("NAT_CI_REDIS_PORT", "6379"))
    db = int(os.environ.get("NAT_CI_REDIS_DB", "0"))
    password = os.environ.get("REDIS_PASSWORD", "redis")
    bucket_name = os.environ.get("NAT_CI_REDIS_BUCKET_NAME", "test")

    try:
        import redis
        client = redis.Redis(host=host, port=port, db=db, password=password)
        if not client.ping():
            raise RuntimeError("Failed to connect to Redis")
        yield {"host": host, "port": port, "db": db, "bucket_name": bucket_name, "password": password}
    except ImportError:
        if fail_missing:
            raise
        pytest.skip("redis not installed, skipping redis tests")
    except Exception as e:
        if fail_missing:
            raise
        pytest.skip(f"Error connecting to Redis server: {e}, skipping redis tests")


@pytest_asyncio.fixture(name="mysql_server", scope="session")
async def fixture_mysql_server(fail_missing: bool) -> AsyncGenerator[dict[str, str | int]]:
    """Fixture to safely skip MySQL based tests if MySQL is not running"""
    host = os.environ.get('NAT_CI_MYSQL_HOST', '127.0.0.1')
    port = int(os.environ.get('NAT_CI_MYSQL_PORT', '3306'))
    user = os.environ.get('NAT_CI_MYSQL_USER', 'root')
    password = os.environ.get('MYSQL_ROOT_PASSWORD', 'my_password')
    bucket_name = os.environ.get('NAT_CI_MYSQL_BUCKET_NAME', 'test')
    try:
        import aiomysql
        conn = await aiomysql.connect(host=host, port=port, user=user, password=password)
        yield {"host": host, "port": port, "username": user, "password": password, "bucket_name": bucket_name}
        conn.close()
    except ImportError:
        if fail_missing:
            raise
        pytest.skip("aiomysql not installed, skipping MySQL tests")
    except Exception as e:
        if fail_missing:
            raise
        pytest.skip(f"Error connecting to MySQL server: {e}, skipping MySQL tests")


@pytest.fixture(name="minio_server", scope="session")
def minio_server_fixture(fail_missing: bool) -> Generator[dict[str, str | int]]:
    """Fixture to safely skip MinIO based tests if MinIO is not running"""
    host = os.getenv("NAT_CI_MINIO_HOST", "localhost")
    port = int(os.getenv("NAT_CI_MINIO_PORT", "9000"))
    bucket_name = os.getenv("NAT_CI_MINIO_BUCKET_NAME", "test")
    aws_access_key_id = os.getenv("NAT_CI_MINIO_ACCESS_KEY_ID", "minioadmin")
    aws_secret_access_key = os.getenv("NAT_CI_MINIO_SECRET_ACCESS_KEY", "minioadmin")
    endpoint_url = f"http://{host}:{port}"

    minio_info = {
        "host": host,
        "port": port,
        "bucket_name": bucket_name,
        "endpoint_url": endpoint_url,
        "aws_access_key_id": aws_access_key_id,
        "aws_secret_access_key": aws_secret_access_key,
    }

    try:
        import botocore.session
        session = botocore.session.get_session()

        client = session.create_client("s3",
                                       aws_access_key_id=aws_access_key_id,
                                       aws_secret_access_key=aws_secret_access_key,
                                       endpoint_url=endpoint_url)
        client.list_buckets()
        yield minio_info
    except ImportError:
        if fail_missing:
            raise
        pytest.skip("aioboto3 not installed, skipping MinIO tests")
    except Exception as e:
        if fail_missing:
            raise
        else:
            pytest.skip(f"Error connecting to MinIO server: {e}, skipping MinIO tests")


@pytest.fixture(name="langfuse_bucket", scope="session")
def langfuse_bucket_fixture(fail_missing: bool, minio_server: dict[str, str | int]) -> Generator[str]:

    bucket_name = os.getenv("NAT_CI_LANGFUSE_BUCKET", "langfuse")
    try:
        import botocore.session
        session = botocore.session.get_session()

        client = session.create_client("s3",
                                       aws_access_key_id=minio_server["aws_access_key_id"],
                                       aws_secret_access_key=minio_server["aws_secret_access_key"],
                                       endpoint_url=minio_server["endpoint_url"])

        buckets = client.list_buckets()
        bucket_names = [b['Name'] for b in buckets['Buckets']]
        if bucket_name not in bucket_names:
            client.create_bucket(Bucket=bucket_name)

        yield bucket_name
    except ImportError:
        if fail_missing:
            raise
        pytest.skip("aioboto3 not installed, skipping MinIO tests")
    except Exception as e:
        if fail_missing:
            raise
        else:
            pytest.skip(f"Error connecting to MinIO server: {e}, skipping MinIO tests")


@pytest.fixture(name="langfuse_url", scope="session")
def langfuse_url_fixture(fail_missing: bool, langfuse_bucket: str) -> str:
    """
    To run these tests, a langfuse server must be running.
    """
    import requests

    host = os.getenv("NAT_CI_LANGFUSE_HOST", "localhost")
    port = int(os.getenv("NAT_CI_LANGFUSE_PORT", "3000"))
    url = f"http://{host}:{port}"
    health_endpoint = f"{url}/api/public/health"
    try:
        response = requests.get(health_endpoint, timeout=5)
        response.raise_for_status()

        return url
    except Exception as e:
        reason = f"Unable to connect to Langfuse server at {url}: {e}"
        if fail_missing:
            raise RuntimeError(reason)
        pytest.skip(reason=reason)


@pytest.fixture(name="langfuse_trace_url", scope="session")
def langfuse_trace_url_fixture(langfuse_url: str) -> str:
    """
    The langfuse_url fixture provides the base url, however the general.telemetry.tracing["langfuse"].endpoint expects
    the trace url which is what this fixture provides.
    """
    return f"{langfuse_url}/api/public/otel/v1/traces"


@pytest.fixture(name="oauth2_server_url", scope="session")
def oauth2_server_url_fixture(fail_missing: bool) -> str:
    """
    To run these tests, an oauth2 server must be running.
    """
    import requests

    host = os.getenv("NAT_CI_OAUTH2_HOST", "localhost")
    port = int(os.getenv("NAT_CI_OAUTH2_PORT", "5001"))
    url = f"http://{host}:{port}"
    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status()

        return url
    except Exception as e:
        reason = f"Unable to connect to OAuth2 server at {url}: {e}"
        if fail_missing:
            raise RuntimeError(reason)
        pytest.skip(reason=reason)


@pytest.fixture(name="oauth2_client_credentials", scope="session")
def oauth2_client_credentials_fixture(oauth2_server_url: str, fail_missing: bool) -> dict[str, typing.Any]:
    """
    Fixture to provide OAuth2 client credentials for testing

    Simulates the steps a user would take in a web browser to create a new OAuth2 client as documented in:
    examples/front_ends/simple_auth/README.md
    """

    try:
        import requests
        from bs4 import BeautifulSoup
        username = os.getenv("NAT_CI_OAUTH2_CLIENT_USERNAME", "Testy Testerson")

        # This post request responds with a cookie that we need for future requests and a 302 redirect, the response
        # for the redirected url doesn't contain the cookie, so we disable the redirect here to capture the cookie
        user_create_response = requests.post(oauth2_server_url,
                                             data=[("username", username)],
                                             headers={"Content-Type": "application/x-www-form-urlencoded"},
                                             allow_redirects=False,
                                             timeout=5)
        user_create_response.raise_for_status()
        cookies = user_create_response.cookies

        client_create_response = requests.post(f"{oauth2_server_url}/create_client",
                                               cookies=cookies,
                                               headers={"Content-Type": "application/x-www-form-urlencoded"},
                                               data=[
                                                   ("client_name", "test"),
                                                   ("client_uri", "https://test.com"),
                                                   ("scope", "openid profile email"),
                                                   ("redirect_uri", "http://localhost:8000/auth/redirect"),
                                                   ("grant_type", "authorization_code\nrefresh_token"),
                                                   ("response_type", "code"),
                                                   ("token_endpoint_auth_method", "client_secret_post"),
                                               ],
                                               timeout=5)
        client_create_response.raise_for_status()

        # Unfortunately the response is HTML so we need to parse it to get the client ID and secret, which are not
        # locatable via ID tags
        soup = BeautifulSoup(client_create_response.text, 'html.parser')
        strong_tags = soup.find_all('strong')
        i = 0
        client_id = None
        client_secret = None
        while i < len(strong_tags) and None in (client_id, client_secret):
            tag = strong_tags[i]
            contents = "".join(tag.contents)
            if client_id is None and "client_id:" in contents:
                client_id = tag.next_sibling.strip()
            elif client_secret is None and "client_secret:" in contents:
                client_secret = tag.next_sibling.strip()

            i += 1

        assert client_id is not None and client_secret is not None, "Failed to parse client credentials from response"

        return {
            "id": client_id,
            "secret": client_secret,
            "username": username,
            "url": oauth2_server_url,
            "cookies": cookies
        }

    except Exception as e:
        reason = f"Unable to create OAuth2 client: {e}"
        if fail_missing:
            raise RuntimeError(reason)
        pytest.skip(reason=reason)


@pytest.fixture(name="local_sandbox_url", scope="session")
def local_sandbox_url_fixture(fail_missing: bool) -> str:
    """Check if sandbox server is running before running tests."""
    import requests
    url = os.environ.get("NAT_CI_SANDBOX_URL", "http://127.0.0.1:6000")
    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        return url
    except Exception:
        reason = (f"Sandbox server is not running at {url}. "
                  "Please start it with: cd src/nat/tool/code_execution/local_sandbox && ./start_local_sandbox.sh")
        if fail_missing:
            raise RuntimeError(reason)
        pytest.skip(reason)


@pytest.fixture(name="sandbox_config", scope="session")
def sandbox_config_fixture(local_sandbox_url: str) -> dict[str, typing.Any]:
    """Configuration for sandbox testing."""
    return {
        "base_url": local_sandbox_url,
        "execute_url": f"{local_sandbox_url.rstrip('/')}/execute",
        "timeout": int(os.environ.get("SANDBOX_TIMEOUT", "30")),
        "connection_timeout": 5
    }


@pytest.fixture(name="piston_url", scope="session")
def piston_url_fixture(fail_missing: bool) -> str:
    """
    Configuration for piston testing.

    The public piston server limits usage to five requests per minute.
    """
    import requests
    url = os.environ.get("NAT_CI_PISTON_URL", "https://emkc.org/api/v2/piston")
    try:
        response = requests.get(f"{url.rstrip('/')}/runtimes", timeout=30)
        response.raise_for_status()
        return url
    except Exception:
        reason = (f"Piston server is not running at {url}. "
                  "Please start it with: cd src/nat/tool/code_execution/local_sandbox && ./start_local_sandbox.sh")
        if fail_missing:
            raise RuntimeError(reason)
        pytest.skip(reason)
